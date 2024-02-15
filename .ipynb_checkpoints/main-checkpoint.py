import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import sys 
sys.path.insert(0,"/home/mtissot/SpinUp/jumper/")
from jump2 import Simulation, Predictions, load_ts
import random 

#term         #Enter the name of the variable 
#simu_path    #Enter the simulation path
#id_          #Select the id - 106 if path6, 104 if path4 - Not really imp anymore
#ye           #Transform monthly simulation to yearly simulation
#start        #Start of the simu : 0 to keep spin up / t to cut the spin up
#end          #End of the simu
#comp         #Explained variance ratio for the pca
#file_save    #File to save prepared simu

def prepare(term,simu_path, id_, start, end, ye, comp, file_save):
    simu = Simulation(path=simu_path,id_=id_,start=start,end=end,ye=ye,comp=comp,term=term)  #Load yearly or monthly simulations                    
    print(f"{term} loaded")                                                                   
    simu.prepare()                     #Prepare simulations : start to end - removeClosedSeas - (removeSSCA) - standardize - to numpy
    print(f"{term} prepared")  
    simu.applyPCA()                    #Exctract time series through PCA 
    print(f"PCA applied on {term}")
    simu_dico = simu.makeDico()        #Create dico: ts - mask - desc -(ssca) - cut(=start) - x_size - y_size - (z_size) - shape
    print(f"{term} to dictionnary")
    if not os.path.exists(file_save):  #save infos
        os.makedirs(file_save)
    with open(file_save + f'pca_{term}', 'wb') as file:
        pickle.dump(simu.pca, file)
    np.savez(file_save + term, **simu_dico)
    print(f"{term} saved")
    del simu

"""
#file    #Select the file where the prepared simu was saved  
#var     #Select the var you want to forecast
def load_ts(file,var):
    dico = np.load(file+f"/{var}.npz",allow_pickle=True)
    dico = {key: dico[key] for key in dico.keys()}
    df   = pd.DataFrame(dico["ts"],columns=[f"{var}-{i+1}" for i in range(np.shape(dico["ts"])[1])])
    with open(file+f"/pca_{var}", 'rb') as file:
        dico["pca"] = pickle.load(file)
    return df,dico
"""

#file_load    #select the file of the prepared simulation   
#term         #select the feature you want to forecast
#id_          #same as in prepare function
#steps        #Number of years you want to forecast
#file_save    #File in which you want to save the predictions

def jump(file_load,term,id_,steps):
    df,infos = load_ts(file_load,term)                                   #load dataframe and infos
    simu_ts = Predictions(id_,term,df,infos)                             #create the class to predict
    print(f"{term} time series loaded") 
    y_hat, y_hat_std, metrics = simu_ts.Forecast(len(simu_ts),steps)     #Forecast
    print(f"{term} time series forcasted") 
    n = len(simu_ts.info["pca"].components_)                             #Reconstruct n predicted components
    predictions_zos     = simu_ts.reconstruct(y_hat,n,begin=len(simu_ts))
    print(f"{term} predictions reconstructed") 
    np.save(f"/data/mtissot/simuPred/predictions{term}.npy", predictions_zos) #Save
    print(f"{term} predictions saved to numpy") 


def emulate(simu_path,steps,id_,ye,start,end,comp,file_simu,file_pred):
    for term in ["zos","so","thetao"]:
        print(f"Preparing {term}...")
        prepare(term,simu_path,id_, ye, start, end, comp, file_simu)
        print()
        print(f"Forecasting {term}...")
        jump(file_simu,term,id_,steps)
        print()

if '__main__':

    #parser = argparse.ArgumentParser(description="Description of your script")
    #parser.add_argument("--ye", type=bool, help="Transform monthly simulation to yearly simulation")
    #parser.add_argument("--start", type=int, help="Start of the simulation")
    #parser.add_argument("--end", type=int, help="End of the simulation")
    #parser.add_argument("--comp", type=float, help="Explained variance ratio for the pca")
    #parser.add_argument("--f", type=str, help="File to save prepared simulation")

    #args = parser.parse_args()
    """
    # LAUNCH FOR SO - THETAO - ZOS
    term      = "so"
    print(f"Preparing {term}...")

    simu_path = "/scratchu/mtissot/SIMUp6Y"
    id_       = "106"                       
    ye        = True                       
    start     = 25                     
    end       = 65                    
    comp      = 0.9                        
    file      = "/data/mtissot/simuPrepared/" 
    prepare(term,simu_path,id_, ye, start, end, comp, file)
    
    print()
    print(f"Forecasting {term}...")
    file_load = file 
    steps     = 30
    file_save = f'/data/mtissot/simuPred/predictions{term}.npy' 
    jump(file,term,id_,steps,file_save)
"""
    simu_path = "/scratchu/mtissot/SIMUp6Y"  
    steps     = 30
    id_       = "106"                       
    ye        = True                       
    start     = 25                     
    end       = 65                    
    comp      = 0.9  
    file_simu = "/data/mtissot/simuPrepared/" 
    emulate(simu_path,steps,id_,ye,start,end,comp,file_simu)
