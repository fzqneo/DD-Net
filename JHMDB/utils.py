import numpy as np
import os
import pandas as pd
import random
import scipy.ndimage.interpolation as inter
from scipy.signal import medfilt 

###################################################################################
    
    
#Rescale/Interpolate to be target_l frames
def zoom(p,target_l=64,joints_num=25,joints_dim=3):
    l = p.shape[0]
    if l == target_l: # need do nothing
        return p
    p_new = np.empty([target_l,joints_num,joints_dim]) 
    for m in range(joints_num):
        for n in range(joints_dim):
#             p_new[:,m,n] = medfilt(p_new[:,m,n],3)  # zf: p_new is uninitialized. this is useless
            p_new[:,m,n] = inter.zoom(p[:,m,n],target_l/l)
    return p_new

def sampling_frame(p,C):
    # randomly sample a subset of the frames of a point, then rescale it to match the original frame length
    full_l = p.shape[0] # full length
    if random.uniform(0,1)<0.5: # aligment sampling
        valid_l = np.round(np.random.uniform(0.85,1)*full_l)
        s = random.randint(0, full_l-int(valid_l))
        e = s+valid_l # sample end point
        p = p[int(s):int(e),:,:]    
    else: # without aligment sampling
        valid_l = np.round(np.random.uniform(0.9,1)*full_l)
        index = np.sort(np.random.choice(range(0,full_l),int(valid_l),replace=False))
        p = p[index,:,:]
    p = zoom(p,C.frame_l,C.joint_n,C.joint_d)
    return p

def norm_scale(x):
    return (x-np.mean(x))/np.mean(x)

from scipy.spatial.distance import cdist
def get_CG(p,C):
    # return JCD of a point, normalized to 0 mean
    M = []
    iu = np.triu_indices(C.joint_n,1,C.joint_n)
    for f in range(C.frame_l): 
        d_m = cdist(p[f],p[f],'euclidean')       
        d_m = d_m[iu] 
        M.append(d_m)
    M = np.stack(M) 
    M = norm_scale(M)
    return M


