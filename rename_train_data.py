import os
dir = r"C:\Users\SHJ\Desktop\sim\dataset3\img_seg"
file_names = os.listdir(dir)
index = 0
for i in file_names:
    j = i.split("_")[-1]
    dst = j.lstrip("0")
    os.rename(dir+"/"+i,dir+"/"+dst)
    print(index)
    index+=1
