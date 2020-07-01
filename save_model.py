#%% 库导入
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten,Dropout,MaxPooling2D
from keras import backend as K
from keras.models import Model
import numpy as np
import os
from keras.applications import MobileNetV2
from keras.applications import VGG19
from keras.applications import VGG16
import cv2
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras import Sequential
from keras.regularizers import l2

#%% 超参数
batch_size = 16
cols = 160
rows = 80
channels = 3
train_dir = "E:/dateset/selfdriving-car-simulator_Uisee1_3/img"
label_dir = "E:/dateset/selfdriving-car-simulator_Uisee1_3/action_pose.txt"
log_dir="log/4_24ep_udacity_3c_kivinet_2pred_test_half/"
model_save_basename = "KivlNet_part2_25_plus"
epochs_initial = 0
epochs_finetune = 5
train_test_ratio = 5
if_save_model_trained = True
save_interval = 1
train_only = False
train_from_pretrained = True
pretrained_model = "log/4_24ep_udacity_3c_kivinet_2pred_test_half/KivlNet_part2_25.h5"
preWhiten = False
speed_angle_weights_ratio = 1 #影响不大
angle_filter_control = True
img_last_name = ".jpg"
if not os.path.isdir("log/"):
    os.mkdir("log/")
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

#%% 构建my model
model = Sequential()
model.add(Conv2D(24, (5, 5),strides=(2,2),activation='relu',input_shape=(rows,cols,channels)))
model.add(BatchNormalization())
model.add(Conv2D(36, (5, 5),strides=(2,2),activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(48, (5, 5),strides=(2,2),activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3),strides=(1,1),activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3),strides=(1,1),activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(100,activation='relu'))
# model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(50,activation='relu'))
# model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(10,activation='relu'))
# model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(2))
# model.compile(loss=keras.losses.MSE,
#               optimizer=keras.optimizers.SGD(lr=1e-3, decay=1e-4, momentum=0.9, nesterov=True),
#               metrics=['mse'])
model.compile(loss=keras.losses.MSE,
              optimizer=keras.optimizers.Adam(lr=0.5*1e-3),
              metrics=['mse'])
model.save('my_model.h5')