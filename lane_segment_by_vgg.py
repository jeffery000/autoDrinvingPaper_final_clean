# -*- coding: utf-8 -*-

#%%
from keras.models import Model
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation
from pylab import *
import keras
from keras.models import load_model
from keras.applications import VGG16
import os
#%%
# original_dir = "/home/stephan/Yushi/dataset/img"
# dst_dir = "/home/stephan/Yushi/dataset/img_seg"
original_dir = "E:/dateset/selfdriving-car-simulator_Uisee_ds1_5/img"
dst_dir = "E:/dateset/selfdriving-car-simulator_Uisee_ds1_5/img_seg"
# input img rows and cols,对于模拟器，是480*320
rows = 160
cols = 320
channels = 3
# 二值化阈值，对于模拟器图片，取8较合适
threshhold = 8
# 卷积输出特征图rows的一半，用来把上半部分去掉，对于模拟器&vgg block3_conv3，取40
half_rows_conv = 20
if not os.path.isdir(dst_dir):
    os.mkdir(dst_dir)
#%% model
model = VGG16(include_top=False,input_shape=(rows,cols,channels))
x = model.get_layer('block3_conv3').output
model = Model(input=model.input, output=x)
model.save("KivlNet_part1.h5")
#%%
#file_names = os.listdir(original_dir)
#for name in file_names:
#    img = cv2.imread(original_dir+"/"+name)
#    img = cv2.resize(img,(cols,rows))
#    img_batch = np.expand_dims(img, axis=0)
#    conv_img = model.predict(img_batch)  # conv_img 卷积结果
#    conv_img_sum = np.sum(conv_img,axis=3)
#    conv_img_sum = np.squeeze(conv_img_sum,axis=0)
#    conv_img_sum = 1/(conv_img_sum/conv_img_sum.max())
#    ret,conv_img_sum=cv2.threshold(np.expand_dims(conv_img_sum,axis=2),threshhold,1,cv2.THRESH_BINARY)
#    conv_img_sum[:half_rows_conv,:] = 0
#    cv2.imwrite(dst_dir+"/"+name,conv_img_sum*255)
#%%   segment byvgg block3   240th feature map 
file_names = os.listdir(original_dir)
file_names.sort(key = lambda x: int(x[:-4]))
# file_names = file_names[:10]
lenth = len(file_names)
count = 0
for name in file_names:
    img = cv2.imread(original_dir+"/"+name)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(cols,rows))
    img_batch = np.expand_dims(img, axis=0)
    conv_img = model.predict(img_batch)  # conv_img 卷积结果
    conv_img_dst = conv_img[0,:,:,240]
    conv_img_dst_nor = conv_img_dst/conv_img_dst.max()*255
    # conv_img_dst_nor[:half_rows_conv,:] = 0
    conv_img_dst_nor = np.array(conv_img_dst_nor,np.uint8)
    cv2.imwrite(dst_dir+"/"+name,conv_img_dst_nor)
    print("processing %s/%s"%(count,lenth))
    count += 1
