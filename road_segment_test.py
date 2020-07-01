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
from matplotlib import pyplot as plt
#%% 超参数
cols = 320
rows = 100
# cols = 1242
# rows = 375
channels = 3

#%% 构建my model
model = load_model("my_seg/my_seg10.h5")

#%%
label_dir = r"D:\dataset\selfdriving-car-simulator_Uisee1_3\img\429.jpg"
# label_dir = r"E:\dateset\selfdriving-car-simulator_Uisee1_3\img\4911.jpg"
# label_dir = r"E:\dateset\comma_c12_uisee_100_102_104_105_115_118_m\img\1157.jpg"
img = cv2.imread(label_dir)
img = cv2.resize(img,(cols,rows))/255

img = np.expand_dims(img,axis=0)
pred = model.predict(img)
pred = np.squeeze(pred,axis=0)
a = np.argmax(pred,axis=2)
# b = pred[:,:,0]
# print("a")
plt.imshow(a)
plt.show()