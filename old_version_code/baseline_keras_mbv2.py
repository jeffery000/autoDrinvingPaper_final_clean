#%% 库导入
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras import backend as K
from keras.models import Model
import numpy as np
import os
from keras.applications import MobileNetV2
from keras.applications import ResNet50
import cv2
#%% 超参数
batch_size = 32
cols = 320
rows = 180
channels = 3
train_dir = "D:\\keyan\\yushikeji\\dataset\\dataset\\train"
#%% 构建网络
mbv2 = MobileNetV2(include_top=False,input_shape=(rows,cols,channels))
x = mbv2.get_layer("Conv_1_bn").output
x = Flatten() (x)
x = Dense(100,activation='relu') (x)
x = Dense(50,activation='relu') (x)
x = Dense(10,activation='relu') (x)
predictions = Dense(2,activation='relu') (x)
mbv2_self_driving = Model(input=mbv2.input, output=predictions)
mbv2_self_driving.save("mbv2_self_driving.h5")
#%% 冻结卷积层
for layer in mbv2_self_driving.layers[:-4]:
    layer.trainable = False
#%% 查看可更新层
for layer in mbv2_self_driving.layers:
    print(layer.trainable)
#%% 求解器设置
mbv2_self_driving.compile(loss=keras.losses.MSLE,
              optimizer=keras.optimizers.adam(),
              metrics=['mse'])
#%% 提取batch
def extract_one_batch(file_names, train_dir, shape_img = [rows,cols,channels]):
    current_batch = np.zeros((len(file_names), shape_img[0], shape_img[1],\
        shape_img[2] ))
    for index_file_name,file_name in enumerate(file_names):
        img = cv2.imread(train_dir+"/"+file_name)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(shape_img[1],shape_img[0]))
        img = img/255
        current_batch[index_file_name] = img
    return current_batch
#%% 训练
file_nams_epoch = os.listdir(train_dir)
label = np.load("D:\\keyan\\yushikeji\\dataset\dataset\\label.npy")
num_batch = len(file_nams_epoch)//batch_size
for batch_id in range(num_batch):
    current_batch = extract_one_batch(file_nams_epoch[batch_id*batch_size:(batch_id+1)*batch_size],\
        train_dir)
    mbv2_self_driving.fit(current_batch, label[batch_id*batch_size:(batch_id+1)*batch_size],\
                    batch_size=batch_size,\
                    epochs=1,\
                    verbose=1)
    if batch_id%1 == 0:
        print("%s st batch is training"%batch_id)

