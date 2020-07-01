import os
import sys
import shutil
src_dir = "E:/dateset/selfdriving-car-simulator/dataset/dataset/"
dst_dir = "E:/dateset/selfdriving-car-simulator_Uisee/"
if not os.path.isdir(dst_dir):
    os.mkdir(dst_dir)
if not os.path.isdir(dst_dir+"img"):
    os.mkdir(dst_dir+"img")
file_names = os.listdir(src_dir+"IMG/")
file_names_center = [i for i in file_names if i[:6]=="center"]

f_label_r = open(src_dir+"driving_log.csv",'r')
f_label_w = open(dst_dir+"action_pose.txt",'w')
f_label_r_lines = f_label_r.readlines()
f_label_r_dict = {}
for i in f_label_r_lines:
	i_split = i.split(",")
	f_label_r_dict[i_split[0].split("\\")[-1]] = i_split[-4]+","+i_split[-1].strip("\n")
len_count = 0
for file_name in file_names_center:
    shutil.copy(src_dir+"/IMG/"+file_name,dst_dir+"img/"+str(len_count)+".jpg")
    f_label_w.write(str(len_count)+","+ file_name +","+f_label_r_dict[file_name]+"\n")
    len_count+=1
    print("process:%dcurrent|%dtotal,percent:%.2f"%(len_count,len(f_label_r_lines),len_count/len(f_label_r_lines)))