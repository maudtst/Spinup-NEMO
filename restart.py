import numpy as np
import xarray as xr
import densite
import matplotlib.pyplot as plt

def getXYslice(array):
    """
    Given a Restart array with 'DOMAIN_position_first' and 'DOMAIN_position_last' attributes,
    this function calculates and returns slices for x and y dimensions.

    Parameters:
        array (xarray.DataArray) : Restart file
        
    Returns
        x_string (slice dtype=float) : range of x positions
        y_string (slice dtype=float) : range of y positions
    """
    First   = array.DOMAIN_position_first  
    Last    = array.DOMAIN_position_last
    x_slice = slice(First[0]-1,Last[0])
    y_slice = slice(First[1]-1,Last[1])
    return x_slice, y_slice


def toXarray(var,name,dep=True,fillna=True):
    """
    Converts a numpy array into an xarray DataArray and replace nan values by 0 to obtain the same data format as in restart files. 

    Parameters:
        var (numpy array)       : The  array to be converted.
        name (str)              : The name to be assigned to the resulting xarray DataArray.
        dep (bool, optional)    : If True, indicates that the array represents dependent variables.
                                   Defaults to True.
        fillna (bool, optional) : If True, fills NaN values with 0 after conversion.
                                   Defaults to True.

    Returns:
        array (xarray.DataArray): An xarray DataArray object representing the input numpy array.
    """
    if dep:
        if len(np.shape(var))==4:
            array = xr.DataArray(var, dims=("time_counter", "nav_lev", "y", "x"), name=name)
        elif len(np.shape(var))==3:
            array = xr.DataArray(var, dims=("nav_lev", "y", "x"), name=name)
    else:
        if len(np.shape(var))==3:
            array = xr.DataArray(var, dims=("time_counter", "y", "x"), name=name)
        elif len(np.shape(var))==2:
            array = xr.DataArray(var, dims=("y", "x"), name=name)
    return array.fillna(0)

def update_pred(array,zos,so,thetao):
    """
    Update the Restart file with the predictions. We use the same prediction step for now and before steps (e.g sshn/sshb). 
    We also update the surface so (sss_m) and thetao (sst_m) 

    Parameters:
        array (xarray.Dataset) : Restart file 
        zos (numpy.ndarray)    : Sea surface height for the current time step    - (t,y,x).
        so (numpy.ndarray)     : Salinity for the current time step              - (t,z,y,x).
        thetao (numpy.ndarray) : Potential temperature for the current time step - (t,z,y,x).

    Returns:
        None
    """
    x_slice,y_slice = getXYslice(array)
    #Changement des variables de restart now 
    array['sshn'] = toXarray(zos[-1:,y_slice,x_slice],"sshn",dep=False)
    array['sn']   = toXarray(so[-1:,:,y_slice,x_slice],"sn")
    array['tn']   = toXarray(thetao[-1:,:,y_slice,x_slice],"tn")
    #Changement des variables de restart before 
    array['sshb'] = toXarray(zos[-1:,y_slice,x_slice],"sshb",dep=False)
    array['sb']   = toXarray(so[-1:,:,y_slice,x_slice],"sb")
    array['tb']   = toXarray(thetao[-1:,:,y_slice,x_slice],"tb")
    #Changement des variables de restart mean
    array['sss_m'] = toXarray(so[-1:,0,y_slice,x_slice],"sss_m",dep=False)
    array['sst_m'] = toXarray(thetao[-1:,0,y_slice,x_slice],"sst_m",dep=False)
    #return array

#E3T_M diff E3T IL Y A BIEN UNE DIMENSION DE PROFONDEUR ?????????.??
#AJOUTER LE CALCUL DE DEPTHT
def update_e3tm(mask_array,array):
    """
    Update e3t_m : the cell thickness of the top layer.
    Get e3t : the cell thickness for all dimensions, we can use e3t to get the new bathymetry and to update u and velocities 
              e3t = e3t_initital * (1+tmask4D*np.expand_dims(np.tile(ssh*ssmask/(bathy+(1-ssmask)),(75,1,1)),axis=0))
    Get deptht : The depth of each cell on grid. we use it to update the density rhop.
    
    Parameters:
        mask_array (xarray.Dataset) : Mask array containing tmask values
        array (xarray.Dataset)      : Restart file

    Returns:
        e3t (numpy.ndarray) : Updated array of z-axis cell thicknesses.
        depth (int)         : The depth value, which is currently set to 0.
    """
    x_slice,y_slice = getXYslice(array)
    e3t_ini = array.variables['e3t_ini']                                        # initial z axis cell's thickness on grid T - (t,z,y,x)
    ssmask  = np.max(mask_array.tmask.values[:,:,y_slice,x_slice],axis=1)       # continent mask                            - (t,y,x)
    bathy   = np.ma.sum(e3t_ini,axis=1)                                         # inital Bathymetry                         - (t,y,x)
    ssh     = array.variables['ssh_m']                                          # Sea Surface Height                        - (t,y,x)
    tmask   = mask_array.tmask.values[:,:,y_slice,x_slice]                      # bathy mask on grid T                      - (t,z,y,x)
    e3t     = e3t_ini*(1+tmask*np.expand_dims(np.tile(ssh*ssmask/(bathy+(1-ssmask)),(75,1,1)),axis=0))  #                   - (t,z,y,x)
    depth=0                                                                    #                                            - (t,z,y,x)
    
    array['e3t_m'] = toXarray(e3t[:,0],"e3t_m",dep=False)  # - (t,y,x)
    #e3t_m    = e3t_ini[:,0]*(1+tmask4D[:,0]*ssh*ssmask/(bathy+(1-ssmask)))    
    #newbathy = np.ma.sum(e3t,axis=1)
    return e3t, depth

def update_rhop(array,thetao,so):
    """
    Update the rhop variable in the array based on temperature (thetao) and salinity (so).

    Parameters:
        array (xarray.Dataset) : Restart file
        thetao (numpy.ndarray) : Temperature predictions
        so (numpy.ndarray)     : Salinity predictions

    Returns:
        None
    """
    x_slice,y_slice = getXYslice(array)
    rhop = densite.sigma_n(thetao[-1:,:,y_slice,x_slice],so[-1:,:,y_slice,x_slice],n=0)    #- (t,z,y,x)
    array['rhop'] = toXarray(rhop,"rhop")


def plot_density_infos(array,e3t_new,min_=1017):
    """
    Plot density (rhop) information: surface density, density as a function of depth, and the difference in density as a function of depth. 
    The difference provides insights into density errors, particularly where it decreases instead of increasing

    Parameters:
        array (xarray.Dataset) : Restart file contains density informations
        e3t_new (numpy.array)  : Array representing the new z-axis cell thickness : the distance between two grid points.
        min_ (float, optional) : Minimum value for color scale. Defaults to 1017.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    a = axes[0].pcolor(array["rhop"][0,0],vmin=min_)
    fig.colorbar(a, ax=axes[0])
    
    rhop = array['rhop'].where(array["rhop"][0] != 0., np.nan)
    diff_rhop = np.diff(rhop.isel(time_counter=0), axis=0) / e3t_new[0,:-1]
    
    for i in range(array["rhop"].sizes['x']): 
        for j in range(array["rhop"].sizes['y']):
            rhop.isel(time_counter=0, x=i, y=j).plot(ax=axes[1])     
            axes[2].plot(diff_rhop[:,j,i])  
            
    axes[0].set_title('Surface density')
    axes[1].set_title('Density as a function of depth')
    axes[2].set_title('Diff Density as a function of depth')


# FONCTIONNE MAIS A VOIR
def regularize_rho(rho):
    """
    Regularize the rho variable to ensure the density is alway superieur or equal at a lower depth. 
    If the value found at k-1 depth is lower than the value found at k. k-1 value is replaces by k value.

    Parameters:
        rho (numpy.ndarray): Array representing density with dimensions (time, depth, latitude, longitude).

    Returns:
        numpy.ndarray : Regularized array of density.
    """
    t,z,y,x = np.shape(rho)
    for i in range(x):
        for j in range(y):
            for k in range(z-1):
                if rho[0,k,j,i]>rho[0,k+1,j,i]:
                    rho[0,k+1,j,i] = rho[0,k,j,i]
    return rho


#ERREYUUU
def update_v_velocity(array,maskarray,e3t_new):  #e3t_new             = maskarray["e3t_0"].values[0,:,y_slice,x_slice]
    """
    Update the v-component velocity array.

    Parameters:
        array (xarray.Dataset)     : Restart file.
        maskarray (xarray.Dataset) : Mask dataset.
        e3t_new (numpy.ndarray)    : Updated array of z-axis cell thicknesses.

    Returns:
        None
    """
    x_slice,y_slice = getXYslice(array)
    vn              = array.variables['vn']                          #initial v velocity of the restart         - (t,z,y,x)
    e1t             = maskarray["e1t"].values[0,y_slice,x_slice]     #initial y axis cell's thickness on grid T - (y,x)
    vmask           = maskarray["vmask"].values[0,:,y_slice,x_slice] #bathy mask on grid V                      - (z,y,x)
    ff_f            = maskarray["ff_f"].values[0,y_slice,x_slice]    #corriolis force                           - (y,x)
    rhop_new        = array.variables['rhop']                        #updated density                           - (t,z,y,x)

    diff_x = -np.roll(rhop_new,shift=1,axis=2) + rhop_new                #                - (t,z,y,x)
    v_new  = 9.81/(rhop_new*ff_f) * np.cumsum(diff_x*e3t_new/e1t,axis=1) # v without V_0  - (t,z,y,x)
    #v_new = add_bottom_velocity(vn.values,v_new.values,vmask)           # add V_0        - (t,z,y,x)
    
    #array['vn']    = toXarray(vn_new,"vn")
    #array['vb']    = toXarray(vn_new,"vb")
    #array['ssv_m'] = toXarray(vn_new[:,0],"vb",dep=False)
    return v_new

def update_u_velocity(array,maskarray,e3t_new):
    """
    Update the v-component velocity array.

    Parameters:
        array (xarray.Dataset)     : Restart file.
        maskarray (xarray.Dataset) : Mask dataset.
        e3t_new (numpy.ndarray)    : Updated array of z-axis cell thicknesses.

    Returns:
        None
    """
    x_slice,y_slice = getXYslice(array) 
    un              = array.variables['un']                          #initial u velocity of the restart         - (t,z,y,x)
    e2t             = maskarray["e2t"].values[0,y_slice,x_slice]     #initial x axis cell's thickness on grid T - (y,x)
    umask           = maskarray["umask"].values[0,:,y_slice,x_slice] #bathy mask on grid U                      - (z,y,x)
    ff_f            = maskarray["ff_f"].values[0,y_slice,x_slice]    #corriolis force                           - (y,x)
    rhop_new        = array.variables['rhop']                        #updated density                           - (t,z,y,x)
    
    diff_y = np.roll(rhop_new,shift=-1,axis=2) - rhop_new                #                - (t,z,y,x)
    u_new  = 9.81/(rhop_new*ff_f) * np.cumsum(diff_y*e3t_new/e2t,axis=1) # u without U_0  - (t,z,y,x)
    #un_new = add_bottom_velocity(un.values,u_new.values,umask)          # add U_0        - (t,z,y,x)
    
    #array['un']    = toXarray(un_new,"un")
    #array['ub']    = toXarray(un_new,"ub")
    #array['ssu_m'] = toXarray(un_new[:,0],"ssu_m",dep=False)
    return u_new

#ERREUR MAUVAIS 
def add_bottom_velocity(v_restart,v_update,mask):
    """
    Add bottom velocity values to the updated velocity array.

    Parameters:
        v_restart (numpy.array) : Restart velocity array                           - (t,z,y,x)
        v_update (numpy.array)  : New velocity array without the initial condition - (t,z,y,x)
        mask (numpy.array)      : Mask array indicating presence of water cells    - (z,y,x)

    Returns:
        v_restart (numpy.array): Velocity array with bottom velocity values added.
    """
    time,deptht,y,x = np.shape(v_update)
    for i in range(x):
        for j in range(y):
            v0=False
            for k in range(deptht)[::-1]:                           # From the bottom to the top :
                if mask[k,j,i]==1 and v0==False:                    #   If first cell of sea in the water column
                    v0 = v_restart[-1,k,j,i]                        #      set V0 to the corresponding value
                elif mask[k,j,i]==1 and v0!=False:                  #   If cell is not in the bottom
                    v_restart[-1,k,j,i] = v0 + v_update[-1,k,j,i]   #      cell is equal to new v cell + v0             
    return v_restart
