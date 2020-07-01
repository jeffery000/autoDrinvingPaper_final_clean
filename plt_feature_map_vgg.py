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

 
    # 各个特征图按1：1 叠加
    plt.figure(1)
    feature_map_sum = sum(ele for ele in feature_map_combination)
    # cv2.imwrite("img.jpg",feature_map_sum)
    plt.imshow(feature_map_sum)
    axis('off')
    plt.savefig("feature_map_sum.png")
 
#%%my_model
#model = load_model("vgg16_angle15.h5")
#x = model.get_layer('block5_conv3').output
#model = Model(input=model.input, output=x)
#%%original vgg16
rows = 80
cols = 160
channels = 3
model = VGG16(include_top=False,input_shape=(rows,cols,channels))
# model.save("vgg_notop.h5")
# model = load_model(r"vgg_jef_lite.h5")
#x = model.get_layer('conv2d_4').output
x = model.get_layer('block2_conv2').output
# x = model.get_layer('lambda_6').output
model = Model(input=model.input, output=x)
model.save("vgg_featur.h5")
#%%
img = cv2.imread(r'E:\dateset\comma_c12_uisee_100_102_104_105_115_118_m\img\6010.jpg')
img = img[80:220,:,:]
img = cv2.resize(img,(cols,rows))/255
img_batch = np.expand_dims(img, axis=0)
conv_img = model.predict(img_batch)  # conv_img 卷积结果

#%%
visualize_feature_map(conv_img)
plt.show()


# %%
