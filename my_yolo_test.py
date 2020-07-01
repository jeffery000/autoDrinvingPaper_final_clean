#%% 库导入
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation,Deconv2D
from keras.layers import AveragePooling2D, Input, Flatten,Dropout,MaxPooling2D,UpSampling2D
from keras import backend as K
from keras.models import Model
import numpy as np
import os
import cv2
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras import Sequential
from keras.utils import to_categorical
import glob
import shutil
from keras.applications import MobileNet
from matplotlib import pyplot as plt
#%% 超参数
cols = 224
rows = 224
channels = 3

#%% 构建my model
model = load_model("my_yolo/my_yolo2.h5")
#%%
label_dir = r"E:\dateset\amateur-unmanned-air-vehicle-detection-dataset\Database1\database2\video16_488.jpeg"
img = cv2.imread(label_dir)
img = cv2.resize(img,(cols,rows))/255

img = np.expand_dims(img,axis=0)
pred = model.predict(img)
pred = np.squeeze(pred,axis=0)
a = np.argmax(pred,axis=2)
# b = pred[:,:,0]
# print("a")
plt.imshow(a)
plt.show()
    
# #%%
# dir2 = r"E:\dateset\amateur-unmanned-air-vehicle-detection-dataset\Database1\Database1"
# img_names = glob.glob(dir2+"/*.txt")
# for i in img_names:
#     f = open(i,"r")
#     l = f.readlines()
#     if len(l) != 0:
#         shutil.copy(i,r"E:\dateset\amateur-unmanned-air-vehicle-detection-dataset\Database1\database2"+"/"+i.split("\\")[-1])
#         shutil.copy(i[:-3]+"jpeg",r"E:\dateset\amateur-unmanned-air-vehicle-detection-dataset\Database1\database2"+"/"+i.split("\\")[-1][:-3]+"jpeg")
# #%%
# dir3 = r"E:\dateset\amateur-unmanned-air-vehicle-detection-dataset\Database1\database2"
# img_name = os.listdir(dir3)[:7]
# rows,cols = [448,448]
# for i in img_name:
#     if i.split(".")[-1]=="jpeg":
#         img = cv2.imread(dir3+"//"+i)
#         img = cv2.resize(img,(rows,cols))
#         f = open(dir3+"//"+i.split(".")[0]+".txt","r")
#         for j in f.readlines():
#             p1,p2,p3,p4 = [float(k) for k in j.split(" ")[1:]]
#             img2 = cv2.rectangle(img,(int((p1-0.5*p3)*cols),int((p2-0.5*p4)*rows)),(int((p1+0.5*p3)*cols),int((p2+0.5*p4)*rows)),color=[255,0,0])
#             cv2.imwrite(dir3+"//"+i.split(".")[0]+".png",img)