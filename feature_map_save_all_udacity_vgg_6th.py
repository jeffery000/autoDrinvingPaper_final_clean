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
original_dir = "D:/dataset/selfdriving-car-simulator_Uisee1_3/img"
dst_dir = "D:/dataset/selfdriving-car-simulator_Uisee1_3/img_vgg_cam_6th"
#%%
def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col
 
 
def visualize_feature_map(img_batch,save_name):
    feature_map = np.squeeze(img_batch, axis=0)
    print(feature_map.shape)
 
    feature_map_combination = []

 
    num_pic = feature_map.shape[2]
    row, col = get_row_col(num_pic)
    # plt.figure(0,figsize=(4*row,4*col))
    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        # plt.subplot(row, col, i + 1)
        # plt.imshow(feature_map_split)
        # axis('off')
    #     title('feature_map_{}'.format(i))
 
    # plt.savefig('feature_map.png')

 
    # 各个特征图按1：1 叠加
    plt.figure(1)
    feature_map_sum = feature_map[:,:,5]
    feature_map_sum = feature_map_sum/(np.max(feature_map_sum))*254
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
rows = 160
cols = 320
channels = 3
model = VGG16(include_top=False,input_shape=(rows,cols,channels))
x = model.get_layer('block3_conv3').output
x = keras.layers.UpSampling2D()(x)
# x = keras.layers.Deconv2D(1,kernel_size=3,kernel_initializer=keras.initializers.Ones())(x)
model = Model(input=model.input, output=x)
#%%
file_names = os.listdir(original_dir)
file_names.sort(key = lambda x: int(x[:-4]))
# file_names = file_names[:10]
# file_names = file_names[:10]
lenth = len(file_names)
count = 0
for name in file_names:
    img = cv2.imread(original_dir+"/"+name)
    img = cv2.resize(img,(cols,rows))/255
    img_batch = np.expand_dims(img, axis=0)
    conv_img = model.predict(img_batch)  # conv_img 卷积结果
    visualize_feature_map(conv_img,dst_dir+"/"+name)
# plt.show()


# %%
