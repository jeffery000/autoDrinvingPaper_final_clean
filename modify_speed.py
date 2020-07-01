# -*- coding: utf-8 -*-
file_path_or = r"C:\Users\SHJ\Desktop\sim\2019.12.3\normal_dataset_henfeijie_back\action_pose.txt"
file_path_dst =  r"C:\Users\SHJ\Desktop\sim\2019.12.3\normal_dataset_henfeijie_back\action_pose2.txt"
f_or = open(file_path_or,'r')
f_dst = open(file_path_dst,'w')
index_list = list(range(232,258))+list(range(1311,1336))+list(range(2394,2421))+list(range(3497,3522))+list(range(4612,4637))
f_or_lines = f_or.readlines()
for index,line in enumerate(f_or_lines):
    if index in index_list:
        f_dst.write(','.join(line.split(',')[:-1])+',8\n')
    else:
        f_dst.write(line)
