import os
import sys
import shutil
src_dir = "D:/dataset/carla527_train2/"
dst_dir = "D:/dataset/carla527_train2_uisee/"
if not os.path.isdir(dst_dir):
    os.mkdir(dst_dir)
if not os.path.isdir(dst_dir+"img"):
    os.mkdir(dst_dir+"img")
file_names = os.listdir(src_dir+"_out/")


f_label_r = open(src_dir+"save_data.txt",'r')
f_label_w = open(dst_dir+"action_pose_carla.txt",'w')
f_label_r_lines = f_label_r.readlines()
slice_list = [int(i*8785/5466) for i in range(5466)]
f_label_r_lines_slice = [f_label_r_lines[i] for i in slice_list]
for lines in f_label_r_lines_slice:
    s = lines.split(",")
    for id,j in enumerate(s):
        if j[:5] =="Speed":
            s2 = j.split(":")[-1][:-5]
            f_label_w.write(s2.strip()+",")
        if j[:8] =="Location":
            s3 = j.split(":")[-1]
            f_label_w.write(s3.strip()+",")
            f_label_w.write(s[id+1]+",")
        if j[:5] =="steer":
            s1 = j.split(":")[-1][:-1]
            f_label_w.write(s1+"\n")

f_label_w.flush()

# f_label_r_dict = {}
# for i in f_label_r_lines:
# 	i_split = i.split(",")
# 	f_label_r_dict[i_split[0].split("\\")[-1]] = i_split[-4]+","+i_split[-1].strip("\n")
len_count = 0
for file_name in file_names:
    shutil.copy(src_dir+"/_out/"+file_name,dst_dir+"img/"+str(len_count)+".jpg")
    # f_label_w.write(str(len_count)+","+ file_name +","+f_label_r_dict[file_name]+"\n")
    len_count+=1
    print("process:%dcurrent|%dtotal,percent:%.2f"%(len_count,len(file_names),len_count/len(file_names)))