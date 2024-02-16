import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, WhiteKernel, DotProduct, ExpSineSquared
import random
from joblib import Parallel, delayed, parallel_backend
import sys 
sys.path.insert(0,"/gpfswork/rech/omr/uen17sn/NewSpinUp/lib2")
#from database 
import prepare as p1
import random
#import warnings
#warnings.filterwarnings("ignore")
#warnings.resetwarnings()

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
        self.var  = var
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
        self.len_ = len(self.data)
        #model param
        #predictions and metrics
        #sef.y_hats, self.metrics, self.y_stds, self.train_lens, self.cy, self.rec_data
    
    def __len__(self):
        return len(self.data)
        
    ###_______predict, show results and get metrics________###  
    
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
        x_pred     = np.arange(0,(len(self)+steps)*pas,pas).reshape(-1,1)
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

    ###________Reconstruct predicitons________###
    
    def reconstruct(self,predictions,n):
        rec = []
        self.int_mask = self.info["mask"].astype(np.int32).reshape(self.info["shape"])
        for t in range(len(predictions)):
            map_ = np.zeros((self.info["shape"]),dtype=np.float32)
            arr  = np.array(list(predictions.iloc[t,:n]) + [0]*(len(self.info["pca"].components_)-n))
            map_[self.int_mask==1] = self.info["pca"].inverse_transform(arr)
            map_[self.int_mask==0] = np.nan
            rec.append(map_)
        return np.array(rec)*2*self.info["desc"]["std"]+self.info["desc"]["mean"]
   
    ###________Get RMSE of the predicitons________###
    
    def getRmseV(self,predictions):
        n = np.product(self.shape) - self.nbNan()
        c = np.sqrt(np.nansum(np.nansum((self.simulation-predictions)**2,axis=-1),axis=-1)/n)
        return c# * 2 * s.desc["std"]

    def rmseMap(self,predictions,train_len):
        t = self.len
        return np.sqrt(np.sum((self.simulation[train_len:]-predictions[train_len:])**2,axis=0)/t) #* 2 * s.desc["std"]

    @staticmethod
    def nbNan(self):
        return np.nansum(np.isnan(self.simulation[0]))
    
    
class optimization:
    
    def __init__(self,ids,ratio,ncomp,var,steps=1,min_test=50,min_train=50,kernels=None,trunc=None):
        random.shuffle(ids)
        i = int(len(ids) * ratio)
        self.ids_eval  = ids[:i]
        self.ids_test  = ids[i:]
        self.simu_eval = [TS(id_,var) for id_ in ids[:i]]
        self.simu_test = [TS(id_,var) for id_ in ids[i:]]
        self.ncomp     = ncomp
        self.var       = var
        self.min_train = min_train
        self.min_test  = min_test
        self.steps     = steps
        self.trunc     = trunc
        self.kernels   = [RBF(), RationalQuadratic()] if kernels is None else kernels
        self.seed      = random.randint(1, 200)
    
    ###___get all gp___###
    
    def getAllGP(self, n=4, r=RBF()):
        kernels = self.kernelCombination(r, n)
        listGP = []
        for kernel in np.array(kernels).reshape(-1):
            kernel = kernel + WhiteKernel()
            listGP.append(GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=0))
        self.gps = listGP

    def kernelCombination(self,r=RBF(), n=4):
        k = self.kernels
        if n == 1:
            return r
        else:
            return (self.kernelCombination(r + k[0], n=n - 1),
                    self.kernelCombination(r * k[0], n=n - 1),
                    self.kernelCombination(r + k[1], n=n - 1),
                    self.kernelCombination(r * k[1], n=n - 1))
    
    ###___evaluate current gp___###
    
    def evaluateCurrentProcess(self):
        random.seed(self.seed)
        results_eval = []
        print("evaluation : ")
        for simu in self.simu_eval:
            print(f"Processing simulation {simu.id}")
            if self.min_train < len(simu)-self.min_test:
                train_lens = np.arange(self.min_train,len(simu)-self.min_test,self.steps)
                results_eval.append(simu.evaluateModel(self.ncomp,train_lens,f"{self.var}-1",jobs=15))
                if self.trunc is None:
                    results_eval[-1] = np.sum(result[-1])
                else:
                    results_eval[-1] = np.sum([min(val, self.trunc) for val in results_eval[-1]])
        results_test = []
        print("\ntest : ")
        for simu in self.simu_test:
            print(f"Processing simulation {simu.id}")
            if self.min_train < len(simu)-self.min_test:
                train_lens = np.arange(self.min_train,len(simu)-self.min_test,self.steps)
                results_test.append(simu.evaluateModel(self.ncomp,train_lens,f"{self.var}-1",jobs=15))
                if self.trunc is None:
                    results_test[-1] = np.sum(results_test[-1])
                else:
                    results_test[-1] = np.sum([min(val, self.trunc) for val in results_test[-1]])
        self.current_score_eval = np.sum(results_eval)
        self.current_score_test = np.sum(results_test)   
    
    ###___evaluate gp___###
    
    #this methode should be changed to rmse with raw simulation
    def evaluateProcess(self,simu,train_lens,process):
        currentgp = simu.gp
        simu.gp   = process
        print("-",end="")
        test = simu.evaluateModel(self.ncomp,train_lens,f"{self.var}-1",jobs=15)
        simu.gp = currentgp
        if self.trunc is None:
            return np.sum(test)
        else : 
            return np.sum([min(val, self.trunc) for val in test])
    
    def evaluateKernels(self):
        random.seed(self.seed)
        results=[]
        for simu in self.simu_eval:
            print(f"Processing simulation {simu.id} ",end="")
            if self.min_train < len(simu)-self.min_test:
                train_lens = np.arange(self.min_train,len(simu)-self.min_test,self.steps)
                results.append([self.evaluateProcess(simu,train_lens,process) for process in self.gps])
                print("",end="\n")
        results = [(process,score) for process,score in zip(self.gps, np.sum(results,axis=0))]
        self.scores_eval = sorted(results, key=lambda item: item[1], reverse=True)
        
    ###___Select on test___###
    
    def testKernels(self):
        random.seed(self.seed)
        gps_test = [process for process,score in self.scores_eval if score > self.current_score_eval]
        results=[]
        for simu in self.simu_test:
            print(f"Processing simulation {simu.id}",end="")
            if self.min_train < len(simu)-self.min_test:
                train_lens = np.arange(self.min_train,len(simu)-self.min_test,self.steps)
                results.append([self.evaluateProcess(simu,train_lens,process) for process in gps_test])
                print("",end="\n")
        results = [(process,score) for process,score in zip(gps_test, np.sum(results,axis=0))]
        self.scores_test = sorted(results, key=lambda item: item[1], reverse=True)
        
    
"""
    def ForecastNoTest(self, comp, steps,  standardize=True):
        x_train = np.arange(0,len(self.data))
        y_train = np.array(self.data.iloc[:,:comp])
        x_pred  = np.arange(len(self.data),len(self.data)+steps) #REMETTRE 0 len(self.data)
        if standardize:
            mean,std,x_train,x_pred,y_train = Predictions.standardize(len(self.data),x_train,y_train,x_pred,None)
        #yt = self.gp.fit(x_train,y_train)  
        self.gp.fit(x_train,y_train)  
        print(self.gp.kernel_)
        y_hat,y_std = self.gp.predict(x_pred,return_std=True) 
        y_hat,y_std = y_hat * 2 * std + mean, y_std * 2 * std
        if comp==1 : 
            y_hat=y_hat.T 
            y_std=y_std.T        
        y_hat  = pd.DataFrame(y_hat, columns=self.data.columns[:comp]).set_index(np.arange(len(self.data),len(self.data)+steps)) #REMETTRE 0 #len(self.data)
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
 """    

"""    
    def reconstructNoTest(self,predictions,n,d=None):
        rec = []
        if d is None:
            self.int_mask = self.info["mask"].astype(np.int32).reshape(self.info["shape"])
       # else:
        #    self.int_mask = self.info["mask"].astype(np.int32).reshape(self.info["z_size"],self.info["y_size"],self.info["x_size"])
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

"""

"""    
    def reconstructPredictions(self,n,jobs=10):
        rec_mean   = Parallel(n_jobs=jobs)(delayed(self.reconstructOnePrediction)(pred,n) for pred in self.y_hats)
        rec_stdUpp = Parallel(n_jobs=jobs)(delayed(self.reconstructOnePrediction)(pred+std,n) for pred,std in zip(self.y_hats,self.y_stds))
        rec_stdLow = Parallel(n_jobs=jobs)(delayed(self.reconstructOnePrediction)(pred-std,n) for pred,std in zip(self.y_hats,self.y_stds))
        return rec_mean, rec_stdUpp, rec_stdLow
"""
"""
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
"""    
"""    #reconstruct test
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
""" 


"""
    def getCY(self,var,out=16,f=2.576,jobs=15):
        self.cy = Parallel(n_jobs=jobs)(delayed(self.countCY)(m,var,out,f) for m in self.metrics)
        
        
    def countCY(self,metric,var,out,f): 
        l = [1 if p < m + f*np.sqrt(std) and p > m - f*np.sqrt(std) else 0 for p,m,std in
             zip(metric[var]["ma_pred"],metric[var]["ma_true"],metric[var]["std_true"])]
        zero = [ind for ind, val in enumerate(l) if val == 0]
        for i in range(len(zero)-out): 
            if (zero[i]+out - zero[i+out]) == 0:
                return len(l[:zero[i]])+self.w//2
        return len(l)+self.w//2
    
    def showCY(self,metric,cy,var,out,f,color):
        y1  = self.w//2
        fig  = plt.figure(figsize=(12,3))  
        axis = np.arange(y1,(len(metric[f"{var}"]["ma_true"])+y1))
        plt.plot(axis, metric[f"{var}"]["ma_true"], linestyle="dashed",color="black",alpha=0.6,label="test")
        plt.fill_between(
            axis,
            metric[f"{var}"]["ma_true"] - f*np.sqrt(metric[f"{var}"]["std_true"]),
            metric[f"{var}"]["ma_true"] + f*np.sqrt(metric[f"{var}"]["std_true"]),
            color = "tab:grey",
            alpha = 0.2)  
        plt.plot(axis, metric[f"{var}"]["ma_pred"], color=color,label="1 eof regression")
        plt.axvline(cy,color = color,linewidth=0.7)
        plt.legend()
        plt.title(f"99% confidence interval of the test serie");
"""  

"""    
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
"""