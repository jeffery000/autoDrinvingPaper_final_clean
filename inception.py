#%%
from keras.models import Model
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D,Deconv2D
from keras.layers import Activation
from pylab import *
import keras
from keras.models import load_model
from keras.applications import VGG16,InceptionV3
def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def normlization(batch):
    if np.ndim(batch)!=4:
        print("ndims is not 4!")
    else:
        num_max = np.max(batch)
        num_min = np.min(batch)
        batch = (batch-num_min)/(num_max-num_min)
    return batch
def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col
def visualize_feature_map(img_batch,fig_name):
    feature_map = np.squeeze(img_batch, axis=0)
    print(feature_map.shape)
    feature_map_combination = []
    num_pic = feature_map.shape[2]
    row, col = get_row_col(num_pic)
    plt.figure(fig_name,figsize=(4*row,4*col))
    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split)
        axis('off')
    plt.savefig(fig_name+".png")
    plt.figure(fig_name+"_sum")
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    axis('off')
    plt.savefig(fig_name+"_sum"+".png")
 
output_layer_name = ["activation_1","activation_2","activation_3","activation_4"]#'activation_1'
img_name = r'E:\dateset\comma_c12_uisee_100_102_104_105_115_118_m\img\6010.jpg'
model = InceptionV3(include_top =False)
x = model.get_layer("mixed0").output
x = keras.layers.Conv2D(64,kernel_size=(1,1)) (x)
model = Model(input=model.input, output=x)

img = cv2.imread(img_name)
img = img[80:220,:,:]
img = cv2.resize(img,(160,80))/255

img_batch = np.expand_dims(img, axis=0)
conv_img1 = model.predict(img_batch)   #1*row*col*32

#%%
visualize_feature_map(conv_img1,"conv1")
# visualize_feature_map(conv_img2,"conv2")
plt.show()


