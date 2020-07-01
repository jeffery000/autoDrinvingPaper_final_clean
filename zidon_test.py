#%%
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras import backend as K
from keras.models import Model,load_model
import numpy as np
import os
from keras.applications import MobileNetV2
from keras.applications import VGG16
import cv2
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

# m = Model()
# train_dir = "/home/stephan/datasets/驭势科技/dataset/dataset/train"
test_dir = r"D:\dataset\yushikeji\test"
m = load_model("vgg16_train_8_full_head_10.h5")
n = load_model("vgg16_seg_15_full_14_tail.h5")
# file_nams_epoch1 = list(range(4760))
# file_nams_epoch2 = list(range(4760,5468))
file_nams_epoch1 = list(range(1043))
file_nams_epoch2 = list(range(1043,1336))
f = open("result.txt",'w')
#%% 输出 前面 部分图片 的预测
for index,file_name in enumerate(file_nams_epoch1):
    img = cv2.imread(test_dir + "/" + str(file_name)+".tiff")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = prewhiten(img)
    img = cv2.resize(img, (320, 180))
    # img = img / 255
    img = np.expand_dims(img,axis=0)
    pred = m.predict(img, batch_size=1, verbose=0)
    pred = np.around(pred, decimals=6)
    f.write(str(file_name)+" "+str(pred[0][0])+" "+str(pred[0][1])+"\n")
    f.flush()
# f.close()
#%% 输出 后面 部分图片 的预测
for index,file_name in enumerate(file_nams_epoch2):
    img = cv2.imread(test_dir + "/" + str(file_name)+".tiff")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = prewhiten(img)
    img = cv2.resize(img, (320, 180))
    # img = img / 255
    img = np.expand_dims(img,axis=0)
    pred = n.predict(img, batch_size=1, verbose=0)
    pred = np.around(pred, decimals=6)
    f.write(str(index+1043)+" "+str(pred[0][0])+" "+str(pred[0][1])+"\n")
    f.flush()
f.close()