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
#%% 超参数
cols = 224
rows = 224
# cols = 1242
# rows = 375
channels = 3

#%% 构建my model
mbv = MobileNet(input_shape=(rows,cols,channels),include_top=False)
out = mbv.output
x = keras.layers.Dense(50,activation="relu") (out)
x = keras.layers.Dense(10,activation="relu") (x)
x = keras.layers.Dense(2,activation="softmax") (x)
model = Model(inputs=mbv.input,outputs=x)
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['mse'])
model.save('my_yolo.h5')
#%%
train_dir = "E:/dateset/amateur-unmanned-air-vehicle-detection-dataset/Database1/database2"
batch_size = 4
epochs = 5
file_nams_label = os.listdir(train_dir)
file_nams_label = [i for i in file_nams_label if i.split(".")[-1]=="jpeg"]


batchs = len(file_nams_label)//batch_size
for epoch in range(epochs):
    for batch_id in range(batchs):
        print("process %d epochs %d batchs"%(epoch+1,batch_id+1))
        data_x = np.zeros([batch_size,rows,cols,channels])
        data_y = np.zeros([batch_size,7,7,2])
        for i in range(batch_size):
            img = cv2.imread(train_dir+"/"+file_nams_label[batch_id*batch_size+i])
            print(file_nams_label[batch_id*batch_size+i])
            data_x[i] = cv2.resize(img,(cols,rows))/255
        for i in range(batch_size):
            f = open(train_dir+"/"+file_nams_label[batch_id*batch_size+i][:-5]+".txt","r")
            label = np.zeros([7,7,2])
            label[:,:,0] = 1
            for j in f.readlines():
                p1,p2,p3,p4 = [float(k) for k in j.split(" ")[1:]]
                label[int(p2*7)-1,int(p1*7)-1] = [0,1]
            data_y[i] = label
        trian_loss_metrics =  model.train_on_batch(x = data_x, y = data_y)
        print("train loss is %f"%trian_loss_metrics[0])
    model.save("my_yolo/my_yolo%d.h5"%(epoch+1))
    
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