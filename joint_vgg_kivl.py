#%%
from keras.models import Model
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Conv2D,Dense,Flatten,MaxPool2D
from keras import layers
from pylab import *
import keras
from keras.models import load_model
from keras.applications import VGG16
from keras import backend as K
import tensorflow as tf
import numpy as np
import os

#%%original vgg16
def slice(x):
    return x[:,:,:,240:241]
def normalize(x):
    y = x/K.max(x)
#    y1 = tf.zeros([1,40,120,1])
#    y2 = y[:,40:,:,:]
#    z = K.concatenate([y1,y2],axis = 1)
    return y
rows = 320
cols = 480
channels = 3
model_vgg = VGG16(include_top=False,input_shape=(rows,cols,channels))
model_kivl = load_model(r"log\11_19_no_abn_vggKivl_model\vgg_kivl_16.h5")
x = model_vgg.get_layer('block3_conv3').output
x = layers.Lambda(slice)(x)
x = layers.Lambda(normalize)(x)
for layer in model_kivl.layers:
    x = layer(x)
#%%
model = Model(input=model_vgg.input, output=x)
model.save("vgg_kivl.h5")
#%%test
rows = 320
cols = 480
channels = 3
img = cv2.imread(r"D:\dataset\yushikeji\no_abn_original\img\8116.png")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_batch = np.expand_dims(img, axis=0)
pred = model.predict(img_batch)
