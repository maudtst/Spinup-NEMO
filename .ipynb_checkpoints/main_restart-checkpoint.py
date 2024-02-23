import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import sys 
import random 
import argparse
sys.path.insert(0,"/home/mtissot/SpinUp/jumper/lib")
import restart
import xarray as xr

def update_restart_slice(restart_file,restart_name,mask_file):
#restart file "/thredds/idris/work/ues27zx/Restarts/"   mask file '/thredds/idris/work/ues27zx/eORCA1.4.2_mesh_mask_modJD.nc'
    """
    Update a restart file with new predictions and related variables.

    Parameters:
        restart_file (str) : Path to the existing restart file.
        restart_name (str) : Name of the restart file.
        file_mask (str)    : Path to the mask file.

    Returns:
        None
    """
    restart_array = xr.open_dataset(restarts_file+restart_name,decode_times=False) #load restart file
    mask_array    = xr.open_dataset(mask_file,decode_times=False)                  #load mask file  
    zos_new,so_new,thetao_new  = restart.load_predictions()                        #load ssh, so and thetao predictions 
    restart.update_pred(restart_array,zos_new,so_new,thetao_new)                   #update restart with ssh, so and thetao predictions
    e3t_new    = restart.update_e3tm(restart_array,mask_array)                     #update e3tm and gete e3t
    deptht_new = restart.get_deptht(restart_array,mask_array)                      #get new deptht for density
    restart.update_rhop(restart_array,mask_array,deptht_new)                       #update density
    restart.update_v_velocity(restart_array,mask_array,e3t_new[0])                 #update meridional velocity
    restart.update_u_velocity(restart_array,mask_array,e3t_new[0])                 #update zonal velocity
    array = array.rename_vars({'xx': 'x','yy':'y'})                                #inverse transformation of x and y vars
    Restart.to_netcdf(restarts_file+restart_name)                                  # save file


#PAs EU LE TEMPS D'ESSAYER 
def update_Restarts(restarts_file,mask_file,jobs=10) :
    """
    Update multiple restart files in parallel.

    Parameters:
        restarts_file (str)  : Path to the directory containing restart files.
        mask_file (str)      : Path to the mask file.
        jobs (int, optional) : Number of parallel jobs to run. default 10.

    Returns:
        None
    """
    restart_names = restart.getRestartFiles(restarts_file)   # SUPER LONG PEUT ETRE LE FAIR EN BASH OU ERREUR
    Parallel(jobs)(delayed(update_restart_slice)(restarts_file,f,mask_file) for file in restart_names))




if __name__ == '__main__':
                                                         
    parser = argparse.ArgumentParser(description="Update of restart files")
    parser.add_argument("--restarts_file",  type=str,   help= "adress of restart files")                     
    parser.add_argument("--mask_file",      type=str,   help= "adress of mask file")                    
    args = parser.parse_args()

    update_Restarts(restarts_file=args.restarts_file,mask_file=args.mask_file)

    #update_restart_files
    
    #python SpinUp/jumper/main/main_restart.py --restarts_file '/thredds/idris/work/ues27zx/eORCA1.4.2_mesh_mask_modJD.nc' --mask_file '/thredds/idris/work/ues27zx/eORCA1.4.2_mesh_mask_modJD.nc'
