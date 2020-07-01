#%%
import numpy as np

# %%
def angle_filter(angle,filter_size = 5):
    filter = np.ones((1,filter_size),dtype=np.float)/5
    for i in range(filter_size//2,angle.shape[0]-filter_size//2):
        mul = np.matmul(filter,angle[i-filter_size//2:i+filter_size//2+1,:])
        angle[i,:] = mul
    return angle


# %%
a = np.array([[0,0,0,0,1,2,1.5,2.4,0,0,1.6,2.5,1.3,0,0,1.5,2.5,2.1]]).T


# %%
angle_filter(a)

# %%
