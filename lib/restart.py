import numpy as np
import xarray as xr
import densite
import matplotlib.pyplot as plt
import copy 
import os


# SUPER LONG PEUT ETRE LE FAIR EN BASH OU ERREUR
def getRestartFiles(path,term):# path = "/thredds/idris/work/ues27zx/Restarts/" term = '19141231'
        """
        Get the restart files of the last simulation step

        Parameters:
            path (str): The path to the restarts files  
            term (str): year of the last simulated step

        Returns:
            list: List of restart files
        """
        grid = []
        for file in os.listdir(path):
            if term+"." in file: 
                grid.append(path+"/"+file)
        return grid


def load_predictions():
    """
    Load predicted data from saved NumPy files.

    Returns:
        zos (numpy.array)    : ssh sea surface height predictions  - (t,y,x)
        so (numpy.array)     : salinity predictions                - (t,z,y,x)
        theato (numpy.array) : temperature predictions             - (t,z,y,x)
    """
    zos    = np.load("/data/mtissot/simus_predicted/zos.npy")       
    so     = np.load("/data/mtissot/simus_predicted/so.npy")       
    thetao = np.load("/data/mtissot/simus_predicted/thetao.npy")    
    return zos[-1:], so[-1:], thetao[-1:]

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


def update_e3tm(array,mask_array):
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
    e3t     = e3t_ini*(1+np.expand_dims(np.tile(ssh*ssmask/(bathy+(1-ssmask)),(75,1,1)),axis=0))#tmask                     - (t,z,y,x)
    e3t     = e3t #+                                        #                            - (t,z,y,x)
    #A COMMENTER 
    array['e3t_m'] = toXarray(e3t[:,0],"e3t_m",dep=False) + array.e3t_ini[:,0]*(1-ssmask)       # - (t,y,x)
    return e3t
    
def get_deptht(array,maskarray):
    """
    Calculate the depth of each vertical level on grid T in the 3D grid.

    Parameters:
        array (xarray.Dataset)     : The dataset containing ocean model variables.
        maskarray (xarray.Dataset) : The dataset containing mask variables.

    Returns:
        deptht (numpy.array) : The depth of each vertical level.
    """
    x_slice,y_slice = getXYslice(array)
    e3w_0 = np.array(maskarray.e3w_0)[:,:,y_slice,x_slice] #initial z axis cell's thickness on grid W - (t,z,y,x)
    e3u_0 = np.array(maskarray.e3u_0)[:,:,y_slice,x_slice] #initial z axis cell's thickness on grid U - (t,z,y,x)
    e3v_0 = np.array(maskarray.e3v_0)[:,:,y_slice,x_slice] #initial z axis cell's thickness on grid V - (t,z,y,x)
    e3t_0 = np.array(maskarray.e3t_0)[:,:,y_slice,x_slice] #initial z axis cell's thickness on grid T - (t,z,y,x)
    tmask = np.array(maskarray.tmask)[:,:,y_slice,x_slice] #grid T continent mask                     - (t,z,y,x)
    ssh   = array.variables['sshn']                        #sea surface height                        - (t,y,x)
    ssmask  = tmask[:,0]                                   #bathymetry                                - (t,y,x)
    bathy   = np.ma.sum(e3t_0,axis=1)                      #initial condition depth 0                 - (t,z,y,x)
    depth_0 = np.zeros(np.shape(e3w_0))                   
    depth_0[:,0] = 0.5 * e3w_0[:,0]
    depth_0[:,1:] = depth_0[:,0] + np.cumsum(e3w_0[:,1:],axis=1)
    deptht = depth_0 * (np.expand_dims(1+ssh/(bathy + 1 - ssmask ),axis=0) * tmask) #depth of each vertical level on grid T - (t,z,y,x)
    return deptht

    
def update_rhop(array,maskarray,deptht):
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
    so     = array['sn'].values 
    thetao = array['tn'].values  
    tmask  = maskarray["tmask"].values[-1:,:,y_slice,x_slice] #bathy mask on grid U          - (z,y,x)

    rhop, rho_insitu = get_density(thetao,so,deptht,tmask)
    array['rhop']    = toXarray(rhop,"rhop")


def get_density(thetao,so,depth,tmask):
    """
    Compute potential density referenced at the surface and density anomaly.

    Parameters:
        thetao (numpy.array) : Temperature array - (t,z,y,x).
        so (numpy.array)     : Salinity array    - (t,z,y,x).
        depth (numpy.array)  : Depth array       - (t,z,y,x).
        tmask (numpy.array)  : Mask array        - (t,z,y,x).

    Returns:
        tuple: A tuple containing:
            array: Potential density referenced at the surface.
            array: Density anomaly.
    """
    rdeltaS = 32.0
    r1_S0  = 0.875/35.16504
    r1_T0  = 1./40.
    r1_Z0  = 1.e-4
    
    EOS000 = 8.0189615746e+02
    EOS100 = 8.6672408165e+02
    EOS200 = -1.7864682637e+03
    EOS300 = 2.0375295546e+03
    EOS400 = -1.2849161071e+03
    EOS500 = 4.3227585684e+02
    EOS600 = -6.0579916612e+01
    EOS010 = 2.6010145068e+01
    EOS110 = -6.5281885265e+01
    EOS210 = 8.1770425108e+01
    EOS310 = -5.6888046321e+01
    EOS410 = 1.7681814114e+01
    EOS510 = -1.9193502195
    EOS020 = -3.7074170417e+01
    EOS120 = 6.1548258127e+01
    EOS220 = -6.0362551501e+01
    EOS320 = 2.9130021253e+01
    EOS420 = -5.4723692739
    EOS030 = 2.1661789529e+01
    EOS130 = -3.3449108469e+01
    EOS230 = 1.9717078466e+01
    EOS330 = -3.1742946532
    EOS040 = -8.3627885467
    EOS140 = 1.1311538584e+01
    EOS240 = -5.3563304045
    EOS050 = 5.4048723791e-01
    EOS150 = 4.8169980163e-01
    EOS060 = -1.9083568888e-01
    EOS001 = 1.9681925209e+01
    EOS101 = -4.2549998214e+01
    EOS201 = 5.0774768218e+01
    EOS301 = -3.0938076334e+01
    EOS401 = 6.6051753097
    EOS011 = -1.3336301113e+01
    EOS111 = -4.4870114575
    EOS211 = 5.0042598061
    EOS311 = -6.5399043664e-01
    EOS021 = 6.7080479603
    EOS121 = 3.5063081279
    EOS221 = -1.8795372996
    EOS031 = -2.4649669534
    EOS131 = -5.5077101279e-01
    EOS041 = 5.5927935970e-01
    EOS002 = 2.0660924175
    EOS102 = -4.9527603989
    EOS202 = 2.5019633244
    EOS012 = 2.0564311499
    EOS112 = -2.1311365518e-01
    EOS022 = -1.2419983026
    EOS003 = -2.3342758797e-02
    EOS103 = -1.8507636718e-02
    EOS013 = 3.7969820455e-01
    
    zh  = depth * r1_Z0                             # depth
    zt  = thetao * r1_T0                           # temperature
    zs  = np.sqrt(np.abs(so + rdeltaS ) * r1_S0 ) # square root salinity
    ztm = tmask
    
    zn3 = EOS013*zt  + EOS103*zs+EOS003
    zn2 = (EOS022*zt    + EOS112*zs+EOS012)*zt  + (EOS202*zs+EOS102)*zs+EOS002
    zn1 = (((EOS041*zt  + EOS131*zs+EOS031)*zt + (EOS221*zs+EOS121)*zs+EOS021)*zt + ((EOS311*zs+EOS211)*zs+EOS111)*zs+EOS011)*zt   + (((EOS401*zs+EOS301)*zs+EOS201)*zs+EOS101)*zs+EOS001
    zn0 = (((((EOS060*zt   + EOS150*zs+EOS050)*zt   + (EOS240*zs+EOS140)*zs+EOS040)*zt  + ((EOS330*zs+EOS230)*zs+EOS130)*zs+EOS030)*zt    + (((EOS420*zs+EOS320)*zs+EOS220)*zs+EOS120)*zs+EOS020)*zt   + ((((EOS510*zs+EOS410)*zs+EOS310)*zs+EOS210)*zs+EOS110)*zs+EOS010)*zt    + (((((EOS600*zs+EOS500)*zs+EOS400)*zs+EOS300)*zs+EOS200)*zs+EOS100)*zs+EOS000
    
    zn  = ( ( zn3 * zh + zn2 ) * zh + zn1 ) * zh + zn0
    
    rhop = zn0 * ztm         # potential density referenced at the surface             
    rho_insitu = zn * ztm      # density anomaly (masked)
    return rhop, rho_insitu


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
    Update the v-component velocity array.Meridional

    Parameters:
        array (xarray.Dataset)     : Restart file.
        maskarray (xarray.Dataset) : Mask dataset.
        e3t_new (numpy.ndarray)    : Updated array of z-axis cell thicknesses.

    Returns:
        None
    """
    x_slice,y_slice = getXYslice(array)
    vn              = array.copy().variables['vn']                   #initial v velocity of the restart         - (t,z,y,x)
    e1t             = maskarray["e1t"].values[0,y_slice,x_slice]     #initial y axis cell's thickness on grid T - (y,x)
    vmask           = maskarray["vmask"].values[0,:,y_slice,x_slice] #bathy mask on grid V                      - (z,y,x)
    ff_f            = maskarray["ff_f"].values[0,y_slice,x_slice]    #corriolis force                           - (y,x)
    tmask           = maskarray["tmask"].values[0,:,y_slice,x_slice]

    rhop_new        = array.variables['rhop'][0]
    rhop_new        = rhop_new.where(tmask).values   #updated density                           - (t,z,y,x)

    diff_x = -np.roll(rhop_new,shift=1,axis=2) + rhop_new                #                - (t,z,y,x)
    v_new  = 9.81/(rhop_new*ff_f) * np.cumsum(diff_x*e3t_new/e1t,axis=1) # v without V_0  - (t,z,y,x)
    v_new  = np.expand_dims(v_new, axis=0)
    vn_new = add_bottom_velocity(vn.values,v_new,vmask)           # add V_0        - (t,z,y,x)

    array['vn']    = toXarray(vn_new,"vn")
    array['vb']    = toXarray(vn_new,"vb")
    array['ssv_m'] = toXarray(vn_new[:,0],"vb",dep=False)
    #return v_new,vn_new

def update_u_velocity(array,maskarray,e3t_new):
    """
    Update the v-component velocity array. Zonal

    Parameters:
        array (xarray.Dataset)     : Restart file.
        maskarray (xarray.Dataset) : Mask dataset.
        e3t_new (numpy.ndarray)    : Updated array of z-axis cell thicknesses.

    Returns:
        None
    """
    x_slice,y_slice = getXYslice(array) 
    un              = array.copy().variables['un']                   #initial u velocity of the restart         - (t,z,y,x)
    e2t             = maskarray["e2t"].values[0,y_slice,x_slice]     #initial x axis cell's thickness on grid T - (y,x)
    umask           = maskarray["umask"].values[0,:,y_slice,x_slice] #bathy mask on grid U                      - (z,y,x)
    ff_f            = maskarray["ff_f"].values[0,y_slice,x_slice]    #corriolis force                           - (y,x)
    rhop_new        = array.variables['rhop']                        #updated density                           - (t,z,y,x)
    tmask           = maskarray["tmask"].values[-1:,:,y_slice,x_slice]
    print(np.shape(tmask))
    print(np.shape(rhop_new))
    rhop_new        = rhop_new.where(tmask).values[0]
    
    diff_y = np.roll(rhop_new,shift=-1,axis=2) - rhop_new                #                - (t,z,y,x)
    u_new  = 9.81/(rhop_new*ff_f) * np.cumsum(diff_y*e3t_new/e2t,axis=1) # u without U_0  - (t,z,y,x)
    u_new  = np.expand_dims(u_new, axis=0)
    un_new = add_bottom_velocity(un.values,u_new,umask)                   # add U_0        - (t,z,y,x)
    

    array['un']    = toXarray(un_new,"un")
    array['ub']    = toXarray(un_new,"ub")
    array['ssu_m'] = toXarray(un_new[:,0],"ssu_m",dep=False)
    #return u_new,un_new

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
