import os
import sys
import shutil
import numpy as np
f_label_r = open("D:/dataset/carla527_test1_train_1_3_m/action_pose_carla.txt",'r')
f_label_w = open("D:/dataset/carla527_test1_train_1_3_m/action_pose_carla_speed.txt",'w')
f_label_r_lines = f_label_r.readlines()
k = 100
def angle_filter(angle,filter_size = 29):
    filter = np.ones((1,filter_size),dtype=np.float)/filter_size
    for i in range(filter_size//2,angle.shape[0]-filter_size//2):
        mul = np.matmul(filter,angle[i-filter_size//2:i+filter_size//2+1,:])
        angle[i,:] = mul
    return np.squeeze(angle,axis=1)
label = np.zeros((len(f_label_r_lines),2))
for i,j in enumerate(f_label_r_lines):
    label[i][0] = float(j.split(",")[-1])
if True:
    label[:,0] = angle_filter(np.reshape(label[:,0],[-1,1]))
label[:,1] =29.5 - 2*np.power(np.abs(label[:,0]*k),0.5)+ 2*np.random.randn(len(label)) #
label[:,1][label[:,1]>29.5] = 29.5
for m,n in enumerate(f_label_r_lines):
    f_label_w.write(n.strip("\n")+","+str(label[m,1])+"\n")
f_label_w.close()
