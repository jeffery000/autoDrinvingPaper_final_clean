# -*- coding: utf-8 -*-

import os
import sys
import shutil
src_dir = "D:/dataset/carla527_test1_train_1_3/"
dst_dir = "D:/dataset/carla527_test1_train_1_3_m/"
if not os.path.isdir(dst_dir):
    os.mkdir(dst_dir)
if not os.path.isdir(dst_dir+"img/"):
    os.mkdir(dst_dir+"img/")
sub_dirs = os.listdir(src_dir)
if(len(sub_dirs)<=1):
    sys.exit("no need to merge")
len_count = 0
f_label = open(dst_dir+"action_pose_carla.txt",'w')
for sub_dir in sub_dirs:
    file_names = os.listdir(src_dir+sub_dir+"/img/")
    for file_name in file_names:
        shutil.copy(src_dir+sub_dir+"/img/"+file_name,dst_dir+"img/"+str(len_count+int(file_name.split(".")[0]))+".jpg")
    f_current_label = open(src_dir+sub_dir+"/action_pose_carla.txt",'r')
    f_label.write(f_current_label.read())
    f_label.flush()
    len_count+=len(file_names)
    