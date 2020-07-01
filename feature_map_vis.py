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
    plt.savefig("feature_map_sum.png")
 
#%%my_model
#model = load_model("vgg16_angle15.h5")
#x = model.get_layer('block5_conv3').output
#model = Model(input=model.input, output=x)
#%%original vgg16
rows = 320
cols = 480
channels = 3
model = VGG16(include_top=False,input_shape=(rows,cols,channels))
# model = load_model(r"vgg_jef_lite.h5")
#x = model.get_layer('conv2d_4').output
x = model.get_layer('block3_conv3').output
# x = model.get_layer('lambda_6').output
model = Model(input=model.input, output=x)
model.save("vgg_featur.h5")
#%%
model = load_model("log/4_12_15ep_comma_3channel_100_102_nvidia_160/KivlNet_part2_10.h5")
x = model.get_layer('conv2d_1').output
# x = model.get_layer('lambda_6').output
model = Model(input=model.input, output=x)
#%%resnet50_v2
rows = 320
cols = 480
channels = 3
model = ResNet50V2(include_top=False,input_shape=(rows,cols,channels))
x = model.get_layer('conv3_block1_2_conv').output
# x = model.get_layer('block3_conv3').output
# x = model.get_layer('lambda_6').output
model = Model(input=model.input, output=x)
# model.save("resnet50_v2.h5")
#%%
# img = cv2.imread(r'D:\dataset\yushikeji\test\0.tiff')
# #img = cv2.resize(img,(320,180))
# img_batch = np.expand_dims(img, axis=0)
# conv_img = model.predict(img_batch)  # conv_img 卷积结果
# conv_img_sum = np.sum(conv_img,axis=3)
# conv_img_sum = np.squeeze(conv_img_sum,axis=0)
# conv_img_sum = 1/(conv_img_sum/conv_img_sum.max())
# ret,conv_img_sum=cv2.threshold(np.expand_dims(conv_img_sum,axis=2),5,1,cv2.THRESH_BINARY)
# conv_img_sum[:40,:] = 0
#%%
img = cv2.imread(r'D:\dataset\comma_c12_uisee_100_102_104_105_115_118_m\img\0.jpg')
img = img/255
img_batch = np.expand_dims(img, axis=0)
conv_img = model.predict(img_batch)  # conv_img 卷积结果
# conv_img_sum = conv_img[0,:,:,240]
#conv_img_sum = np.squeeze(conv_img_sum,axis=0)
#conv_img_sum = 1/(conv_img_sum/conv_img_sum.max())
#ret,conv_img_sum=cv2.threshold(np.expand_dims(conv_img_sum,axis=2),8,1,cv2.THRESH_BINARY)
#conv_img_sum[:45,:] = 0
# cv2.imwrite("1573518191670290_0000005753.jpg",conv_img_sum*255)
#conv_img_sum[conv_img_sum>12] =100
#%%
# plt.imshow(conv_img_sum)#='Greys')
plt.show()
#%%
visualize_feature_map(conv_img)



# %%
