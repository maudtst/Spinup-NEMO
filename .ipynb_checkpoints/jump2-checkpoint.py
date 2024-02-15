import os
import copy
import random
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.decomposition
from sklearn.decomposition import PCA
import joblib
from joblib import Parallel, delayed, parallel_backend
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, WhiteKernel, DotProduct#, Matérn
import pickle
import warnings
warnings.filterwarnings("ignore")


#file    #Select the file where the prepared simu was saved  
#var     #Select the var you want to forecast
def load_ts(file,var):
    dico = np.load(file+f"/{var}.npz",allow_pickle=True)
    dico = {key: dico[key] for key in dico.keys()}
    df   = pd.DataFrame(dico["ts"],columns=[f"{var}-{i+1}" for i in range(np.shape(dico["ts"])[1])])
    with open(file+f"/pca_{var}", 'rb') as file:
        dico["pca"] = pickle.load(file)
    return df,dico


class Simulation:
    
    ##################################
    ##                              ##
    ##   LOAD AND PREPARE A SIMU    ##
    ##                              ##
    ##################################
    
    def __init__(self,id_,path,term,start=0,end=None,comp=0.9,ye=True,ssca=False): #choose jobs 3 if 2D else 1
        self.path  = path
        self.id    = id_
        self.term  = term
        self.files = Simulation.getData(path,term)
        self.start = start
        self.end   = end
        self.ye    = ye
        self.comp  = comp
        self.len   = 0
        self.desc  = {}
        self.getAttributes()       #self time_dim, y_size, x_size,
        self.getSimu()             #self simu , desc {"mean","std","min","max"} 

    #### Load files and dimensions info ###
    
    def getData(path,term):
        grid = []
        for file in sorted(os.listdir(path)):
            if term+"." in file: #add param!=""
                grid.append(path+"/"+file)
        return grid
    
    def getAttributes(self):
        array = xr.open_dataset(self.files[0], decode_times=False,chunks={"time": 200, "x":120})
        self.time_dim  = list(array.dims)[3]
        self.y_size    = array.sizes['y']
        self.x_size    = array.sizes['x']
        if "deptht" in array[self.term].dims:
            self.z_size = array.sizes['deptht']
            self.shape = (self.z_size,self.y_size,self.x_size)
        elif "olevel" in array[self.term].dims:
            self.z_size = array.sizes['olevel']
            self.shape = (self.z_size,self.y_size,self.x_size)
        else:
            self.z_size = None
            self.shape = (self.y_size,self.x_size)
    
    #### Load simulation ###
        
    def getSimu(self):
        #array = list(Parallel(jobs)(delayed(self.loadFile)(file) for file in self.files))
        array = [self.loadFile(file) for file in self.files if self.len<self.end]
        array = xr.concat(array, self.time_dim)
        self.desc = {"mean":np.nanmean(array),"std":np.nanstd(array),"min":np.nanmin(array),"max":np.nanmax(array)} 
        self.simulation = array
       
    def loadFile(self,file):
        array = xr.open_dataset(file, decode_times=False,chunks={"time": 200, "x":120})
        array = array[self.term]
        #if self.ye:
        #    #array = array.coarsen({self.time_dim: 12}).mean()   #TO CHANGE WITH TIME DIM
        if self.len + array.sizes[self.time_dim] > self.end:
            array = array[0:self.end-self.len]
            self.len = self.len + array.sizes[self.time_dim]
        else:
            self.len = self.len + array.sizes[self.time_dim]
        return array.load()
    

    #########################
    #  prepare simulation   #
    #########################
    
    def prepare(self):
        if self.end is not None:
            self.simulation = self.simulation[self.start:self.end]
        else:
            self.simulation = self.simulation[self.start:]
        self.len = np.shape(self.simulation)[0] 
        self.removeClosedSeas()
        if self.ye==False:
            self.removeSSCA()
        self.desc.update({"mean":np.nanmean(self.simulation),"std":np.nanstd(self.simulation),
                          "min":np.nanmin(self.simulation),"max":np.nanmax(self.simulation)})
        self.standardize() 
        self.simulation = self.simulation.values
        
    def removeSSCA(self):
        array = self.simulation
        ssca  = np.array(array).reshape((self.len//12, 12)+ self.shape) #np.array(array[self.term])
        ssca  = np.mean(ssca, axis=0)
        ssca_extended = np.tile(ssca, (self.len//12, 1, 1)) 
        self.desc["ssca"] = ssca
        self.simulation = array - ssca_extended
        
            
    def removeClosedSeas(self):
        array   = self.simulation
        y_range = [slice(240, 266),slice(235, 276),slice(160, 201)]  #mer noir, grands lacs, lac victoria 
        x_range = [slice(195, 213),slice(330, 351),slice(310, 325)] 
        for y,x in zip(y_range,x_range):
            array = array.where((array['x'] < x.start) | (array['x'] >= x.stop) | 
                                (array['y'] < y.start) | (array['y'] >= y.stop),drop=True)
        self.simulation = array
        
    def standardize(self):
        self.simulation = (self.simulation - self.desc["mean"]) / (2*self.desc["std"]) 
        
    ##################
    #  Compute PCA   #
    ##################
        
    def applyPCA(self):
        array = self.simulation.reshape(self.len,-1)
        self.bool_mask = np.asarray(np.isfinite(array[0,:]), dtype=bool)
        array_masked = array[:,self.bool_mask]
        pca = PCA(self.comp, whiten=False)
        self.components = pca.fit_transform(array_masked)
        self.pca  = pca
        
    def getPC(self,n):
        map_ = np.zeros((np.product(self.shape)), dtype=float)
        map_[~self.bool_mask] = np.nan
        map_[self.bool_mask]  = self.pca.components_[n]
        map_ = map_.reshape(self.shape)
        map_ = 2 * map_ * self.desc["std"] + self.desc["mean"] 
        return map_

    ##################
    #   Save in db   #
    ##################
    
    def makeDico(self):
        dico = dict()
        dico["ts"]    = self.components.tolist()
        dico["mask"]  = self.bool_mask
        dico["desc"]  = self.desc
        dico["ssca"]  = 0
        dico["cut"]   = self.start
        dico["x_size"]= self.x_size
        dico["y_size"]= self.y_size
        if self.z_size is not None:
            dico["z_size"]= self.z_size
        dico["shape"]= self.shape
        return dico


class Predictions:

    #################################
    ##                             ##
    ##   Forecast & reconstruct    ##
    ##                             ##
    #################################
    
    
    def __init__(self,id_,var,data=None,info=None,gp=None,w=12):
        self.id   = id_
        self.var  = var
        self.gp   = Predictions.defineGP() if gp is None else gp
        self.w    = w
        self.data = data
        self.info=info
        self.info["desc"] = self.info["desc"].item()
        self.len_ = len(self.data)
    
    def __len__(self):
        return len(self.data)
        
    ##################
    #    Forecast    #
    ##################
    
    @staticmethod
    def defineGP():
        long_term_trend_kernel =  0.1*DotProduct(sigma_0=0.0) #+ 0.5*RBF(length_scale=1/2)# +
        irregularities_kernel  = 10 * ExpSineSquared(length_scale=5/45, periodicity=5/45)#0.5**2*RationalQuadratic(length_scale=5.0, alpha=1.0) + 10 * ExpSineSquared(length_scale=5.0)
        noise_kernel           = 2*WhiteKernel(noise_level=1)#0.1**2*RBF(length_scale=0.01) + 2*WhiteKernel(noise_level=1)
        kernel =   irregularities_kernel + noise_kernel +long_term_trend_kernel
        return GaussianProcessRegressor(kernel=kernel, normalize_y=False,n_restarts_optimizer=8)
     

    def Forecast(self,train_len,steps,jobs=1):
        r = Parallel(n_jobs=jobs)(delayed(self.forecast_ts)(c,train_len,steps) for c in range(1,self.data.shape[1]+1))
        y_hats = pd.DataFrame(np.array([r[i][0] for i in range(len(r))]).T, columns=self.data.columns) 
        y_stds = pd.DataFrame(np.array([r[i][1] for i in range(len(r))]).T, columns=self.data.columns) 
        metrics = [r[i][2] for i in range(len(r))]  
        return y_hats, y_stds, metrics
    
    def forecast_ts(self,n,train_len,steps=0):
        random.seed(20)
        mean,std,y_train,y_test,x_train,x_pred = self.prepare(n,train_len,steps) 
        self.gp.fit(x_train,y_train)  
        y_hat,y_hat_std = self.gp.predict(x_pred,return_std=True) 
        y_train,y_hat,y_hat_std = y_train*2*std+mean, y_hat*2*std+mean, y_hat_std*2*std
        metrics = None
        if y_test is not None:
            metrics = Predictions.getMetrics(2,y_hat[train_len:len(self)],y_test)
        return y_hat,y_hat_std,metrics

    def prepare(self,n,train_len,steps):
        x_train    = np.linspace(0,1,train_len).reshape(-1,1)
        pas        = x_train[1,0]-x_train[0,0]
        x_pred = np.arange(0,(len(self)+steps)*pas,pas).reshape(-1,1)
        y_train    = self.data[self.var+"-"+str(n)].iloc[:train_len].to_numpy()
        mean, std  = np.nanmean(y_train), np.nanstd(y_train)
        y_train    = (y_train-mean)/(2.0*std)
        y_test     = None
        if train_len<len(self):
            y_test = self.data[self.var+"-"+str(1)].iloc[train_len:len(self)].to_numpy()
        return mean,std,y_train,y_test,x_train,x_pred

    def show(self,n,y_hat,y_hat_std,train_len,color="tab:blue"):
        figure = plt.figure(figsize=(10,4))
        plt.plot(self.data[self.var+"-"+str(n)][:train_len], linestyle="dashed", color="black", alpha=0.7, label = "Train serie")
        if(train_len < len(self)):
             plt.plot(self.data[self.var+"-"+str(n)][train_len-1:], linestyle="dashed", color="black", alpha =0.5,label = "Test serie")
        plt.plot(y_hat,color=color, label = f"GP forecast")
        plt.fill_between(np.arange(0,len(y_hat)),y_hat+y_hat_std, y_hat-y_hat_std, color = color,alpha = 0.2) 
        plt.title(f"Forecast of {self.var} {str(n)}")
        plt.legend()
        plt.show()  
        print()
 
    @staticmethod
    def getMetrics(w,y_hat,y_test):
        ma_test = np.convolve(y_test/w,              np.ones(w), mode="valid")
        ma_pred = np.convolve(y_hat /w,              np.ones(w), mode="valid")
        dist    = np.convolve((y_hat-y_test)/w,      np.ones(w), mode="valid")
        mse     = np.convolve(((y_hat-y_test)**2)/w, np.ones(w), mode="valid")
        dist_max,std=[],[]
        for i in range(w,len(y_hat)+1):
            windowT = y_test[i-w:i]
            windowH = y_hat[i-w:i]
            #maxi/mini
            maxi = np.max(windowT)-np.mean(windowT)
            mini = np.mean(windowT)-np.min(windowT)
            dist_max.append(max(maxi,mini))
            #std
            std.append(np.std(windowT,ddof=1))
        return {"ma_true":ma_test,"ma_pred":ma_pred,"dist":dist, "dist_max":dist_max,"mse":mse,"std_true":np.array(std)}

    ###################
    #   Reconstruct   #
    ###################
    
    def reconstruct(self,predictions,n,begin=0):
        rec = []
        self.int_mask = self.info["mask"].astype(np.int32).reshape(self.info["shape"])
        for t in range(begin,len(predictions)):
            map_ = np.zeros((self.info["shape"]),dtype=np.float32)
            arr  = np.array(list(predictions.iloc[t,:n]) + [0]*(len(self.info["pca"].components_)-n))
            map_[self.int_mask==1] = self.info["pca"].inverse_transform(arr)
            map_[self.int_mask==0] = np.nan
            rec.append(map_)
        return np.array(rec)*2*self.info["desc"]["std"]+self.info["desc"]["mean"]
   
 
