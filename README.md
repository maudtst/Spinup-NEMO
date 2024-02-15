
# Restart.ipynb
### *Update of restart files for NEMO* 

### Maskdataset  
- dimensions :  t:1 y:331 x:360 z:75  
- umask : continent mask for u grid (continent : 0, sea : 1)  
- vmask : continent mask for v grid (continent : 0, sea : 1)
- e3t_0
- e2t
- e1t 

### Predictions  
- zos    : sea surface height (ssh) - t,y,x  
- so     : salinity - t,z,y,x  
- thetao : temperature - t,z,y,x
- depth  : Not predicted - thickness of the z coordinates

### Restart file  

There is a total of 340 restart file per year. Each file contains a slice of x and y dimensions.   
58 data variables which N are updates using the predictions
ssh, u, b, t, s : now and before 
sss_m, sst_m

#variables de restart now   
sshn : zos at last step  
Restart['sn']    = xarray_so[-1:,:,y_slice,x_slice]  
Restart['tn']    = xarray_thetao[-1:,:,y_slice,x_slice]  
Restart['sss_m'] = xarray_so[-1:,0,y_slice,x_slice]  
Restart['sst_m'] = xarray_thetao[-1:,0,y_slice,x_slice]  



## 3 
rho insitu

