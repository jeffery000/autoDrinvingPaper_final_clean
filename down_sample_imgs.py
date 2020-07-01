import os
import sys
import shutil
src_dir = "D:/dataset/carla527_train2_uisee/"
dst_dir = "D:/dataset/carla527_train2_uisee1_3/"
if not os.path.isdir(dst_dir):
    os.mkdir(dst_dir)
if not os.path.isdir(dst_dir+"img"):
    os.mkdir(dst_dir+"img")
file_names = os.listdir(src_dir+"img/")
file_names.sort(key = lambda x: int(x[:-4]))
f_label_r = open(src_dir+"action_pose_carla.txt",'r')
f_label_w = open(dst_dir+"action_pose_carla.txt",'w')
f_label_r_lines = f_label_r.readlines()
len_count = 0
len_count2 = 0
for file_name in file_names:
    if len_count%3 == 0:
        shutil.copy(src_dir+"/img/"+file_name,dst_dir+"img/"+str(len_count2)+".jpg")
        f_label_w.write(f_label_r_lines[len_count])
        len_count2+=1
    len_count+=1
    print("process:%dcurrent|%dtotal"%(len_count,len(f_label_r_lines)))
