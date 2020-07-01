import numpy as np
import os
import cv2
data_path = r"F:\Datasets\YUSHI\train"
save_path = r"F:\Datasets\YUSHI\generate_img"
label_list = np.load("./label.npy")
num_img = len(os.listdir(data_path))
label_generate = []
flag =0
for img_name in range(4760,num_img-1):
    fore_img = cv2.imread(data_path+"/"+str(img_name)+".tiff")
    rear_img = cv2.imread(data_path+"/"+str(img_name+1)+".tiff")
    target_img = cv2.addWeighted(fore_img,0.5,rear_img,0.5,0)
    cv2.imwrite(save_path+"/"+str(num_img+flag)+".tiff",target_img)
    fore_label = label_list[img_name]
    rear_label = label_list[img_name+1]
    target_label = np.add(fore_label,rear_label)*0.5
    label_generate.append(target_label)
    flag +=1
np.save("./label_generate.npy",label_generate)