from keras.models import Model
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D,Dense,MaxPooling2D,Deconv2D,UpSampling2D,BatchNormalization
from keras.layers import Activation,Lambda,Concatenate
from pylab import *
import keras
from keras.models import load_model
from keras.applications import VGG16
from keras import backend as K
from keras import layers
import os
def mean(x):
    x = keras.backend.mean(x,axis=-1)
    return keras.backend.expand_dims(x,axis=3)

rows = 80
cols = 160
channels = 3
# model = VGG16(include_top=False,input_shape=(rows,cols,channels))
# x1 = model.get_layer('block1_conv2').output #block1_conv2  block2_conv2 block3_conv3
# x2 = model.get_layer('block2_conv2').output
# x3 = model.get_layer('block3_conv3').output
# x1 = keras.layers.MaxPool2D() (x1)
# x3 = keras.layers.UpSampling2D() (x3)
# x3 = keras.layers.Deconv2D(1,kernel_size=3,padding="same",kernel_initializer=keras.initializers.Ones()) (x3)
# x1 = Lambda(mean,name="average1")(x1)
# x2 = Lambda(mean,name="average2")(x2)
# x3 = Lambda(mean,name="average3")(x3)


# x_fla_input = MaxPooling2D()(model.input)
# x_fla_con = keras.layers.Concatenate ()([x1,x2,x3])
# # x_input = keras.layers.Input([160,320,6])
# # x_fla_con = layers.Conv2D(32,(5,5),padding="same",activation="relu")(x_input)
# # x_fla_con = layers.MaxPooling2D()(x_fla_con)
# # x_fla_con = layers.Conv2D(32,(5,5),padding="same",activation="relu")(x_fla_con)
# # x_fla_con = layers.MaxPooling2D()(x_fla_con)
# # x_fla_con = layers.Conv2D(64,(3,3),padding="same",activation="relu")(x_fla_con)
# # x_fla_con = layers.Conv2D(64,(3,3),padding="same",activation="relu")(x_fla_con)
# # x_fla_con = layers.MaxPooling2D()(x_fla_con)
# # x_fla_con = layers.Dense(500,activation='relu')(x_fla_con)
# # x_fla_con = layers.Dense(100,activation='relu')(x_fla_con)
# # x_fla_con = layers.Dense(20,activation='relu')(x_fla_con)
# # x_fla_con = layers.Dense(2)(x_fla_con)
# model_all = Model(input=model.input, output=x_fla_con)

model = Sequential()
model.add(Conv2D(32, (3, 3),padding="same",input_shape=(rows,cols,channels)))
model.add(Conv2D(64, (3, 3),padding="same",activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3, 3),padding="same"))
model.add(Conv2D(128, (3, 3),padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (3, 3),padding="same"))
model.add(Conv2D(256,(3, 3),padding="same"))
model.add(Deconv2D(128,(3,3),padding="same"))
model.add(BatchNormalization())
model.add(UpSampling2D())
model.add(Deconv2D(128,(3,3),padding="same"))
model.add(Deconv2D(64,(3,3),padding="same"))
model.add(BatchNormalization())
model.add(UpSampling2D())
model.add(Deconv2D(64,(3,3),padding="same"))
model.add(Deconv2D(2,(3,3),padding="same"))

model.save("my_seg_paperPlot.h5")