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
#%% 超参数
cols = 320
rows = 100
# cols = 1242
# rows = 375
channels = 3

#%% 构建my model
model = Sequential()
model.add(Conv2D(32, (3, 3),padding="same",input_shape=(rows,cols,channels)))
model.add(Conv2D(64, (3, 3),padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3, 3),padding="same"))
model.add(Conv2D(128, (3, 3),padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (3, 3),padding="same"))
model.add(Conv2D(256,(3, 3),padding="same"))
model.add(Activation("relu"))
model.add(Deconv2D(128,(3,3),padding="same"))
model.add(BatchNormalization())
model.add(UpSampling2D())
model.add(Deconv2D(128,(3,3),padding="same"))
model.add(Deconv2D(64,(3,3),padding="same"))
model.add(BatchNormalization())
model.add(UpSampling2D())
model.add(Deconv2D(64,(3,3),padding="same"))
model.add(Dense(128,activation="softmax"))
model.add(Dense(10,activation="softmax"))
model.add(Dense(2,activation="softmax"))
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['mse'])
model.save('my_segmnt.h5')
if True:
    model = load_model("my_seg/my_seg10.h5")
#%%
train_dir = r"E:/dateset/roadlane-detection-evaluation-2013/data_road/data_road/training/image_2/"
label_dir = "E:/dateset/roadlane-detection-evaluation-2013/data_road/data_road/training/gt_image_2/"
batch_size = 4
epochs = 10
file_nams_label = os.listdir(label_dir)
file_names_label_road = [i for i in file_nams_label if i.split('_')[1]=="road"]
file_names_img_road = os.listdir(train_dir)

batchs = len(file_names_label_road)//batch_size
for epoch in range(epochs):
    for batch_id in range(batchs):
        print("process %d epochs %d batchs"%(epoch+1,batch_id+1))
        data_x = np.zeros([batch_size,rows,cols,channels])
        data_y = np.zeros([batch_size,rows,cols,2])
        for i in range(batch_size):
            img = cv2.imread(train_dir+file_names_img_road[batch_id*batch_size+i])
            data_x[i] = cv2.resize(img,(cols,rows))/255
        for i in range(batch_size):
            temp = cv2.imread(label_dir+file_names_label_road[i])
            temp = cv2.resize(temp,(cols,rows))
            temp = np.sum(temp,axis=2)
            temp2 = np.zeros([rows,cols,2])
            temp2[:,:] = [1,0]
            temp2[temp>300] = [0,1]
            data_y[i] = temp2
        trian_loss_metrics =  model.train_on_batch(x = data_x, y = data_y)
        print("train loss is %f"%trian_loss_metrics[0])
    model.save("my_seg/my_seg10_plus%d.h5"%(epoch+1))
    
