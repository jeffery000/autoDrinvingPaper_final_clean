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
original_dir = "D:/dataset/comma_c12_uisee_100_102_104_105_115_118_m/img"
dst_dir = "D:/dataset/comma_c12_uisee_100_102_104_105_115_118_m/img_seg_cam"
#%%
def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col
 
 
def visualize_feature_map(img_batch,save_name):
    # feature_map = np.squeeze(img_batch, axis=0)
    feature_map = img_batch
    print(feature_map.shape)
 
    feature_map_combination = []

 
    # num_pic = feature_map.shape[2]
    # row, col = get_row_col(num_pic)
    # # plt.figure(0,figsize=(4*row,4*col))
    # for i in range(0, num_pic):
    #     feature_map_split = feature_map[:, :, i]
    #     feature_map_combination.append(feature_map_split)
        # plt.subplot(row, col, i + 1)
        # plt.imshow(feature_map_split)
        # axis('off')
    #     title('feature_map_{}'.format(i))
 
    # plt.savefig('feature_map.png')

 
    # 各个特征图按1：1 叠加
    plt.figure(1)
    feature_map_sum = sum(ele for ele in feature_map_combination)
    # feature_map_sum = 255- np.max(feature_map_sum) + feature_map_sum
    cv2.imwrite(save_name,feature_map_sum)
    # plt.imshow(feature_map_sum)
    # axis('off')
    # plt.savefig(save_name)
 
#%%my_model
#model = load_model("vgg16_angle15.h5")
#x = model.get_layer('block5_conv3').output
#model = Model(input=model.input, output=x)
#%%original vgg16
rows = 100
cols = 320
channels = 3
model = load_model("my_seg/my_seg10.h5")

#%%
file_names = os.listdir(original_dir)
file_names.sort(key = lambda x: int(x[:-4]))
# file_names = file_names[:10]
# file_names = file_names[:10]
lenth = len(file_names)
count = 0
for name in file_names:
    img = cv2.imread(original_dir+"/"+name)
    img = img[80:220,:,:]
    img = cv2.resize(img,(cols,rows))/255
    img_batch = np.expand_dims(img, axis=0)
    conv_img = model.predict(img_batch)  # conv_img 卷积结果
    conv_img = np.squeeze(conv_img,axis=0)
    conv_img = np.argmax(conv_img,axis=2)
    conv_img = conv_img*254 
    cv2.imwrite(dst_dir+"/"+name,conv_img)
    # visualize_feature_map(conv_img,dst_dir+"/"+name)
# plt.show()


# %%
