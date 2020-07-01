import os
import shutil
src_dir = r"C:\Users\SHJ\Desktop\sim\2019.12.10\auto_save_data_back\img"
dst_dir = src_dir+"_index_reset"
#src_f = open(r"C:\Users\SHJ\Desktop\sim\2019.12.10\auto_save_data_back\action_pose.txt",'r')
#dst_f = open(r"C:\Users\SHJ\Desktop\sim\2019.12.10\auto_save_data_back\action_pose2.txt",'w')
#src_f_lines = src_f.readlines()
src_file_names = os.listdir(src_dir)
src_file_names.sort(key = lambda x: int(x[:-4]))
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)
count = 0
lenth = len(src_file_names)
for file_name in src_file_names:
    shutil.copy(src_dir+"/"+file_name,dst_dir+"/"+str(count)+".png")
#    dst_f.write(src_f_lines[int(file_name.split(".")[0])]+"\n")
    count += 1
    print("processing %s/%s"%(str(count),lenth))
