
# Restart.ipynb
### *Update of restart files for NEMO* 

### Maskdataset  
  
dimensions :  t:1 y:331 x:360 z:75  
umask : continent mask for u grid (continent : 0, sea : 1)  
vmask : continent mask for v grid (continent : 0, sea : 1)  

### Predictions  
zos    : sea surface height (ssh) - t,y,x  
so     : salinity - t,z,y,x  
thetao : temperature - t,z,y,x  

### Restart  

340 file per year sliced per range of x and y  
Restart.DOMAIN_position_first  #zone geographique du fichier de restart  
Restart.DOMAIN_position_last  

#variables de restart now   
sshn : zos at last step  
Restart['sn']    = xarray_so[-1:,:,y_slice,x_slice]  
Restart['tn']    = xarray_thetao[-1:,:,y_slice,x_slice]  
Restart['sss_m'] = xarray_so[-1:,0,y_slice,x_slice]  
Restart['sst_m'] = xarray_thetao[-1:,0,y_slice,x_slice]  



## 3 
rho insitu

