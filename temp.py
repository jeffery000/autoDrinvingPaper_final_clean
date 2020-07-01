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
def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col
 
 
def visualize_feature_map(img_batch):
    feature_map = np.squeeze(img_batch, axis=0)
    print(feature_map.shape)
 
    feature_map_combination = []

 
    num_pic = feature_map.shape[2]
    row, col = get_row_col(num_pic)
    plt.figure(0,figsize=(4*row,4*col))
    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split)
        axis('off')
    #     title('feature_map_{}'.format(i))
 
    plt.savefig('feature_map.png')
    # plt.show()
 
    # 各个特征图按1：1 叠加
    plt.figure(1)
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    axis('off')
    plt.savefig("feature_map_sum.png",dpi=500,bbox_inches = 'tight')
# original_dir = "/home/stephan/Yushi/dataset/img"
# dst_dir = "/home/stephan/Yushi/dataset/img_seg"
original_dir = "E:/dateset/comma_c12_uisee_100_102/img"
dst_dir = "E:/dateset/comma_c12_uisee_100_102/img_seg"
# input img rows and cols,对于模拟器，是480*320
rows = 80
cols = 160
channels = 3
# 二值化阈值，对于模拟器图片，取8较合适
threshhold = 8
# 卷积输出特征图rows的一半，用来把上半部分去掉，对于模拟器&vgg block3_conv3，取40
half_rows_conv = 40
if not os.path.isdir(dst_dir):
    os.mkdir(dst_dir)
#%% model
model = VGG16(include_top=False,input_shape=(rows,cols,channels))
x = model.get_layer('block1_conv2').output #block1_conv2  block2_conv2 block3_conv3
x = keras.layers.MaxPool2D() (x)
# x = keras.layers.UpSampling2D() (x)
# x = keras.layers.Deconv2D(1,kernel_size=3,kernel_initializer=keras.initializers.Ones()) (x)
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

img = cv2.imread(r'E:\dateset\selfdriving-car-simulator_Uisee1_3\img\556.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# img = img[80:220,:,:]
img = cv2.resize(img,(cols,rows))
img_batch = np.expand_dims(img, axis=0)
conv_img = model.predict(img_batch)  # conv_img 卷积结果
visualize_feature_map(conv_img)
plt.show()

