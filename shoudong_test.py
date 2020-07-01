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
# test_read_index = np.load('./test_index.npy')
# label = np.load('./label.npy')
# m = Model()
test_dir = "/home/stephan/datasets/驭势科技/dataset/dataset/test"
m = load_model("vgg16_12.h5")
test_read_index = np.arange(len(os.listdir(test_dir)))
loss = np.empty((len(test_read_index),3))
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
for i in range(len(test_read_index)):
    value = []
    img_path = test_dir+'/'+str(test_read_index[i])+'.tiff'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = prewhiten(img)
    img = cv2.resize(img, (320, 180))
    # img = img / 255
    img = np.expand_dims(img,axis=0)
    pred = m.predict(img, batch_size=1, verbose=0)
    loss[i,0] = test_read_index[i]
    loss[i,1] = pred[0,0]
    loss[i, 2] = pred[0,1]
with open('./result.txt','w') as file:
    for i in range(len(loss)):
        line = str(int(loss[i,0]+1))+' '+ "%.6f" % loss[i,1] + ' ' + "%.6f" % loss[i,2] + '\n'
        file.write(line)
print(pred)