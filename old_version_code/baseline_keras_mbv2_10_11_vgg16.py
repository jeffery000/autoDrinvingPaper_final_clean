#%% 库导入
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras import backend as K
from keras.models import Model
import numpy as np
import os
from keras.applications import MobileNetV2
from keras.applications import VGG16
import cv2
#%%
# mbv2 = MobileNetV2(include_top=True)
# mbv2.save("mbv2.h5")
#%% 超参数
batch_size = 64
cols = 320
rows = 180
channels = 3
# train_dir = "/home/stephan/datasets/驭势科技/dataset/dataset/train"
train_dir = "/home/stephan/datasets/驭势科技/dataset/dataset/train"
epochs_initial = 1
epochs_finetune = 5
train_test_ratio = 5
#%% 构建网络
mbv2 = VGG16(include_top=False,input_shape=(rows,cols,channels))
# x = mbv2.get_layer("Conv_1_bn").output
x = mbv2.output

x = AveragePooling2D() (x)
x = Flatten() (x)
# x = Dense(100,activation='relu') (x)
x = Dense(500,activation='relu') (x)
x = Dense(100,activation='relu') (x)
predictions = Dense(2) (x)
mbv2_self_driving = Model(input=mbv2.input, output=predictions)
mbv2_self_driving.save("mbv2_self_driving.h5")
#%% 冻结卷积层
for layer in mbv2_self_driving.layers[:-3]:
    layer.trainable = False
#%% 查看可更新层
# for layer in mbv2_self_driving.layers:
#     print(layer.trainable)
#%% 求解器设置
mbv2_self_driving.compile(loss=keras.losses.MSE,
              optimizer=keras.optimizers.adam(),
              metrics=['mse'])
#%% 提取batch
def extract_one_batch(list_train_index, train_dir, shape_img = [rows,cols,channels]):
    current_batch = np.zeros((len(list_train_index), shape_img[0], shape_img[1],\
        shape_img[2]))
    for index_file_name,file_name in enumerate(list_train_index):
        img = cv2.imread(train_dir+"/"+str(file_name)+".tiff")
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(shape_img[1],shape_img[0]))
        img = img/255
        current_batch[index_file_name] = img
    return current_batch
#%% 训练1
file_nams_epoch = os.listdir(train_dir)
num_total_file = len(file_nams_epoch)
label = np.load("./label.npy")
#%% 划分测试集 + 打乱数据
train_read_index = np.arange(num_total_file)
test_read_index = [i*train_test_ratio for i in list(range(num_total_file//train_test_ratio))]
train_read_index = [j for j in train_read_index if j not in test_read_index]
np.random.shuffle(train_read_index)
num_batch = len(train_read_index)//batch_size
num_test_batch = len(test_read_index)//batch_size
label_test = label[test_read_index]
label_train = label[train_read_index]

#%% 训练2
for epoch in range(epochs_initial+epochs_finetune):
    if epoch >= epochs_initial:
        for layer in mbv2_self_driving.layers[-5:]:
            layer.trainable = True
    for layer in mbv2_self_driving.layers:
        print(layer.trainable)
    for batch_id in range(num_batch):
        current_batch = extract_one_batch(train_read_index[batch_id*batch_size:(batch_id+1)*batch_size],\
            train_dir)
        mbv2_self_driving.fit(x = current_batch, y = label_train[batch_id*batch_size:(batch_id+1)*batch_size],\
                        batch_size=batch_size,\
                        epochs=1,\
                        verbose=1)
        if batch_id%5 == 0:
            print("%s epoch is training"%(epoch+1))
            print("%s st batch is training"%(batch_id+1))
        if batch_id%10 == 0 and batch_id>0:
                for batch_test_id in range(num_test_batch):
                    current_batch = extract_one_batch(test_read_index[batch_test_id*batch_size:(batch_test_id+1)*batch_size],\
                        train_dir)
                    pred = mbv2_self_driving.predict(current_batch,\
                                    batch_size=batch_size,\
                                    verbose=1)
                    loss = np.mean(np.power(np.sum(np.power((pred - label_test[batch_test_id*batch_size:(batch_test_id+1)*batch_size]),2),axis = 1),0.5))
                    print(loss)
    if epoch%2 == 0:
        mbv2_self_driving.save("vgg16"+str(epoch)+".h5")


