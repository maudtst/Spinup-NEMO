import numpy as np

def add_bottom_velocity(v_restart,v_update,mask):
    time,deptht,y,x = np.shape(v_restart)
    for i in range(x):
        for j in range(y):
            v0=False
            for k in range(deptht)[::-1]:                        # From the bottom to the top :
                if mask[-1,k,j,i]==1 and v0==False:              #   If first cell of sea in the water column
                    v0 = v_restart[-1,k,j,i]                     #      set V0 to the corresponding value
                elif mask[-1,k,j,i]==1 and v0!=False:            #   If cell is not in the bottom
                    v_restart[-1,k,j,i] = v0 + v_update[k,j,i]   #      cell is equal to new v cell + v0     
    return v_restart

#   If sea and V0 is already set  #and np.isnan(v[k-1,j,i])==False: