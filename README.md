
# Restart.ipynb
### *Update of restart files for NEMO* 

### Maskdataset  
- dimensions : t:1 y:331 x:360 z:75  
- umask : continent mask for u grid (continent : 0, sea : 1)  
- vmask : continent mask for v grid (continent : 0, sea : 1)
- e3t_0
- e2t
- e1t 

### Features  
- zos        \t: Predicted sea surface height (ssh) - t,y,x  
- so         : Predicted salinity - t,z,y,x  
- thetao     : Predicted temperature - t,z,y,x
- depth      : thickness of the z coordinates
- rho        :
- rho_insitu : density

### Restart file  

There is a total of 340 restart file per year. Each file contains a slice of x and y dimensions.   
58 data variables which N are updates using the predictions
Now and before (n/b) :
- ssh   :  sea surface height       => last prediction of zos
- s     :  sea salinity             => last prediction of so
- t     :  sea temperature          => last prediction of thetao
- sss_m :  sea surface salinity     => last prediction of so
- sst_m :  sea surface temperature  => last prediction of thetao
- v     :  zonal velocity           => 
- u     :


