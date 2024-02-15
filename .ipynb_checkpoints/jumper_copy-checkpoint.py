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
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, WhiteKernel #, Matérn
import pickle
import warnings
warnings.filterwarnings("ignore")


#import sys 
#sys.path.insert(0,"/gpfswork/rech/omr/uen17sn/NewSpinUp/lib2")
#from database 
#import prepare as p1


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
        if self.ye:
            array = array.coarsen({self.time_dim: 12}).mean()   #TO CHANGE WITH TIME DIM
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

    ######################################################################
    
    def rmseOfPCA(self,n):
        reconstruction = self.reconstruct(n)
        rmse_values    = self.rmseValues(reconstruction)*2*self.desc["std"]
        rmse_map       = self.rmseMap(reconstruction)   *2*self.desc["std"]
        return reconstruction, rmse_values, rmse_map
    
    def reconstruct(self, n):
        rec = []
        #int_mask =   # Convert the boolean mask to int mask once
        self.int_mask = self.bool_mask.astype(np.int32).reshape(self.shape)       # Reshape to match the shape of map_
        for t in range(len(self.components)):
            map_ = np.zeros(self.shape, dtype=np.float32)
            arr = np.array(list(self.components[t, :n]) + [0] * (len(self.pca.components_) - n))
            map_[self.int_mask==1] = self.pca.inverse_transform(arr)
            map_[self.int_mask==0] = np.nan
            rec.append(map_)
        return np.array(rec)

    def rmseValues(self,reconstruction):
        n = np.product(self.shape) - self.nbNan()
        return  np.sqrt(np.nansum(np.nansum((self.simulation[:]-reconstruction)**2,axis=-1),axis=-1)/n)

    def rmseMap(self,reconstruction):
        t = self.len
        return np.sqrt(np.sum((self.simulation[:]-reconstruction)**2,axis=0)/t)

    def nbNan(self):
        return np.sum(self.int_mask==False)
    
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
    
    def save(self,file):
        simu_dico = np.load(file+".npz",allow_pickle=True)
        simu_dico = {key: simu_dico[key] for key in simu_dico.keys()}
        for var in simu_dico: simu_dico[var] = simu_dico[var].item()
        simu_dico[self.id] = self.makeDico()
        np.savez(file, **simu_dico)
        with open(f"/gpfswork/rech/omr/uen17sn/NewSpinUp/datasets/pca/simu{self.id}", 'wb') as file:
            pickle.dump(self.pca, file)  

            

def loadInfo(id_):
    dico = np.load(f"/gpfswork/rech/omr/uen17sn/NewSpinUp/datasets/simu_dico.npz",allow_pickle=True)
    dico = {key: dico[key] for key in dico.keys()}
    for var in dico: dico[var] = dico[var].item()
    return dico[id_] 

def createDataFrame(id_,var):
    dico = loadInfo(id_)
    array = np.array(dico["ts"])
    names = [f"{var}-{i+1}" for i in range(np.shape(array)[1])]
    df = pd.DataFrame(array,columns=names)
    return df

def loadPCA(id_):
    with open(f"/gpfswork/rech/omr/uen17sn/NewSpinUp/datasets/pca/simu{id_}", 'rb') as file:
        return pickle.load(file)
       
    
class Predictions:
    
    def __init__(self,id_,var,data=None,info=None,gp=None,w=12):
        #simulation cara
        self.id   = id_
        self.gp   = Predictions.defineGP() if gp is None else gp
        self.w    = w
        if data is None:
            self.data = createDataFrame(id_,var)
        else : 
            self.data = data
        if info is None :
            self.info = loadInfo(self.id)
            self.info["pca"] = loadPCA(self.id)
            #self.info["desc"] = self.info["desc"].item()
        else:
            self.info=info
        self.info["desc"] = self.info["desc"].item()
        #model param
        #predictions and metrics
        #sef.y_hats, self.metrics, self.y_stds, self.train_lens, self.cy, self.rec_data
    
    def __len__(self):
        return len(self.data)
        
    ###_______predict, show results and get metrics________###  
    
    @staticmethod
    def defineGP():
        long_term_trend_kernel = 50.0**2*RBF(length_scale=100.0)
        irregularities_kernel  = 0.5**2*RationalQuadratic(length_scale=5.0, alpha=1.0)
        noise_kernel           = 0.1**2*RBF(length_scale=0.01) + 2*WhiteKernel(noise_level=1)
        kernel = long_term_trend_kernel + irregularities_kernel + noise_kernel
        return GaussianProcessRegressor(kernel=kernel, normalize_y=False,n_restarts_optimizer=1)
      
    def ForecastNoTest(self, comp, steps,  standardize=True):
        x_train = np.arange(0,len(self.data))
        y_train = np.array(self.data.iloc[:,:comp])
        x_pred  = np.arange(len(self.data),len(self.data)+steps)
        if standardize:
            mean,std,x_train,x_pred,y_train = Predictions.standardize(len(self.data),x_train,y_train,x_pred,None)
        self.gp.fit(x_train,y_train)   
        y_hat,y_std = self.gp.predict(x_pred,return_std=True) 
        y_hat,y_std = y_hat * 2 * std + mean, y_std * 2 * std
        if comp==1 : 
            y_hat=y_hat.T 
            y_std=y_std.T        
        y_hat  = pd.DataFrame(y_hat, columns=self.data.columns[:comp]).set_index(np.arange(len(self.data),len(self.data)+steps)) 
        y_std  = pd.DataFrame(y_std, columns=self.data.columns[:comp]).set_index(np.arange(len(self.data),len(self.data)+steps)) 
        return y_hat,y_std
    
    def Forecast(self, comp, train_lens,jobs=20):
        self.train_lens = train_lens
        r = Parallel(n_jobs=jobs)(delayed(self.getOneForecast)(comp,t) for t in train_lens)
        self.y_hats  = [r[i][0] for i in range(len(r))]
        self.y_stds   = [r[i][1] for i in range(len(r))]
        self.metrics = [r[i][2] for i in range(len(r))]
    
    def getOneForecast(self, comp, train_len):
        mean,std,x_train,y_train,x_test,y_test = self.prepare(comp,train_len)
        self.gp.fit(x_train,y_train)   
        y_hat,y_std = self.gp.predict(x_test,return_std=True) 
        y_hat,y_std = y_hat * 2 * std + mean, y_std * 2 * std
        if comp==1 : 
            y_hat=y_hat.T 
            y_std=y_std.T
        y_hat  = pd.DataFrame(y_hat, columns=self.data.columns[:comp]).set_index(np.arange(train_len,len(self.data))) 
        y_std  = pd.DataFrame(y_std, columns=self.data.columns[:comp]).set_index(np.arange(train_len,len(self.data))) 
        metrics={}
        for var in y_hat.columns:
            metrics[var] = Predictions.getMetrics(self.w,y_hat[var].values, self.data[var].values[train_len:])
        return y_hat,y_std,metrics
     
    def prepare(self, comp, train_len,standardize=True):
        #define train & test
        x_train = np.arange(0,train_len)
        y_train = np.array(self.data.iloc[:train_len,:comp])
        x_test  = np.arange(train_len,len(self.data))
        y_test  = np.array(self.data.iloc[train_len:,:comp])
        if standardize:
            return Predictions.standardize(train_len,x_train,y_train,x_test,y_test)
        else:
            return x_train,y_train,x_test,y_test        
    
    @staticmethod
    def standardize(train_len,x_train,y_train,x_test,y_test):
        mean = np.nanmean(y_train, axis=None, keepdims=True)
        std  = np.nanstd (y_train, axis=None, keepdims=True)
        x_train   = x_train.reshape(-1,1)/train_len
        x_test    = x_test.reshape(-1,1) /train_len
        y_train   = (y_train - mean) / (2.0 * std)
        if y_test is not None:
            y_test    = (y_test  - mean) / (2.0 * std)
            return mean,std,x_train,y_train,x_test,y_test
        else:
            return mean,std,x_train,x_test,y_train
    
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
    
    def showOneForecast(self,train_len,y_hat,y_std,var="zos-1",color="tab:blue"):
        figure = plt.figure(figsize=(10,4))
        if train_len is not None:
            plt.plot(self.data[var][:train_len], linestyle="dashed", color="black", alpha=0.8, label = "Train serie")
            plt.plot(self.data[var][train_len:], linestyle="dashed", color="black",  alpha =0.5,label = "Test serie")
        else:
            plt.plot(self.data[var][:], linestyle="dashed", color="black", alpha=0.8, label = "Train serie")
            #plt.plot(self.data[var][train_len:], linestyle="dashed", color="black",  alpha =0.5,label = "Test serie")
            #plt.plot(y_hat[var],color=color, label = f"GP forecast")
        plt.plot(y_hat[var],color=color, label = f"GP forecast")
        plt.fill_between(y_hat.index, y_hat[var]+y_std[var], y_hat[var]-y_std[var], color = color,alpha = 0.2) 
        plt.title(f"Forecasting {var}")
        plt.legend()
        plt.show()
        
    
    
        
    ###________Reconstruct predicitons________###
    
    def reconstructNoTest(self,predictions,n,d=None):
        rec = []
        if d is None:
            self.int_mask = self.info["mask"].astype(np.int32).reshape(self.info["y_size"],self.info["x_size"])
        else:
            self.int_mask = self.info["mask"].astype(np.int32).reshape(self.info["z_size"],self.info["y_size"],self.info["x_size"])
        for t in range(len(predictions)):
            if d is None:
                map_ = np.zeros((self.info["y_size"],self.info["x_size"]),dtype=np.float32)
            else:
                map_ = np.zeros((self.info["z_size"],self.info["y_size"],self.info["x_size"]),dtype=np.float32)
            arr  = np.array(list(predictions.iloc[t,:n]) + [0]*(len(self.info["pca"].components_)-n))
            map_[self.int_mask==1] = self.info["pca"].inverse_transform(arr)
            map_[self.int_mask==0] = np.nan
            rec.append(map_)
        return np.array(rec)

    
    def reconstructPredictions(self,n,jobs=10):
        rec_mean   = Parallel(n_jobs=jobs)(delayed(self.reconstructOnePrediction)(pred,n) for pred in self.y_hats)
        rec_stdUpp = Parallel(n_jobs=jobs)(delayed(self.reconstructOnePrediction)(pred+std,n) for pred,std in zip(self.y_hats,self.y_stds))
        rec_stdLow = Parallel(n_jobs=jobs)(delayed(self.reconstructOnePrediction)(pred-std,n) for pred,std in zip(self.y_hats,self.y_stds))
        return rec_mean, rec_stdUpp, rec_stdLow
        
    def reconstructOnePrediction(self,prediction,n):
        prediction = np.array(prediction)
        rec = []
        for t in range(np.shape(prediction)[0]): 
            map_ = np.zeros((self.info["y_size"],self.info["x_size"]),dtype=np.float32)
            arr  = np.array(list(prediction[t,:n]) + [0]*(len(self.info["pca"].components_)-n))
            map_[np.array(self.info["mask"])]  = self.info["pca"].inverse_transform(arr)
            map_[~np.array(self.info["mask"])] = np.nan  
            map_ = map_ * 2 * self.info["desc"]["std"] + self.info["desc"]["mean"] 
            rec.append(map_)  
        return np.array(rec)
    
    #reconstruct test
    def reconstruct(self,n):
        rec  = []
        for t in range(len(self.data)): 
            map_ = np.zeros((self.info["y_size"],self.info["x_size"]),dtype=np.float32)
            arr  = np.array(list(self.data.iloc[t,:n].values) + [0]*(len(self.info["pca"].components_)-n))
            map_[np.array(self.info["mask"])]  = self.info["pca"].inverse_transform(arr)
            map_[~np.array(self.info["mask"])] = np.nan  
            map_ = map_ * 2 * self.info["desc"]["std"] + self.info["desc"]["mean"] 
            rec.append(map_)  
        self.rec_data = rec
    
    ###________Get RMSE of the predicitons________###
    
    def getOneRmse(self,prediction,train_len):
        return Predictions.rmseValues(prediction,np.array(self.rec_data)[train_len:])
        
    def rmseValues(a,b):
        n = np.shape(a)[1] * np.shape(a)[2] - Predictions.nbNan(a[0])
        return  np.sqrt(np.nansum(np.nansum((a-b)**2,axis=1),axis=1)/n) #*2*info["std"]

    def rmseMap(a,b):
        t = np.shape(a)[0]
        return np.sqrt(np.sum((a-b)**2,axis=0)/t) #*2*#self.info["std"]
            
    def absoluteValues(a,b):
        n = np.shape(a)[1] * np.shape(a)[2] - Predictions.nbNan(a[0])
        return np.nansum(np.nansum(np.abs(a-b),axis=1),axis=1)/n

    def absoluteMap(a,b):
        t = np.shape(a)[0]
        return np.sum(np.abs(a-b),axis=0)/t

    def nbNan(carte):
        return np.sum(np.isnan(carte))
    
    