#%% 库导入
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten,Dropout,MaxPooling2D
from keras import backend as K
from keras.models import Model
import numpy as np
import os
from keras.applications import MobileNetV2
from keras.applications import VGG19
from keras.applications import VGG16
import cv2
from keras.models import load_model
from keras import callbacks
#%% 超参数
batch_size = 64
cols = 320
rows = 180
channels = 3
# train_dir =  r"F:\Datasets\YUSHI\train"
train_dir = r"D:\dataset\yushikeji\train"
epochs_initial = 1
epochs_finetune = 15
train_test_ratio = 5
save_interval = 2
train_only = True
train_from_pretrained = True
pretrained_model = "vgg16_seg_15_15.h5"
trained_by_head = False
preWhiten = True
#%% 构建网络
mbv2 = VGG16(include_top=False,input_shape=(rows,cols,channels))
# x = mbv2.get_layer("Conv_1_bn").output
x = mbv2.output
x = MaxPooling2D() (x)
x = Flatten() (x)
x = Dense(500,activation='relu') (x)
# x = Dropout(0.7) (x)
x = Dense(100,activation='relu') (x)
# x = Dropout(0.7) (x)
predictions = Dense(2) (x)
mbv2_self_driving = Model(input=mbv2.input, output=predictions)
mbv2_self_driving.save("vgg16_self_driving.h5")
#%% 冻结卷积层
for layer in mbv2_self_driving.layers[:-3]:
    layer.trainable = False
# 查看可更新层
# for layer in mbv2_self_driving.layers:
#     print(layer.trainable)
#%% 求解器设置
mbv2_self_driving.compile(loss=keras.losses.MSE,
              optimizer=keras.optimizers.adam(),
              metrics=['mse'])
#%%读取模型
if train_from_pretrained:
    mbv2_self_driving = load_model(pretrained_model)
#%% 提取batch
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
def extract_one_batch(list_train_index, train_dir, shape_img = [rows,cols,channels]):
    current_batch = np.zeros((len(list_train_index), shape_img[0], shape_img[1],\
        shape_img[2]))
    for index_file_name,file_name in enumerate(list_train_index):
        img = cv2.imread(train_dir+"/"+str(file_name)+".tiff")
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if preWhiten:
            img = prewhiten(img)
        img = cv2.resize(img,(shape_img[1],shape_img[0]))
        # img = img/255
        current_batch[index_file_name] = img
    return current_batch

#%% 训划分测试集 + 打乱数据   前面的模型
if trained_by_head:
    file_nams_epoch = os.listdir(train_dir)[:4760]
    num_total_file = len(file_nams_epoch)
    label = np.load("./label.npy")
    # train_read_index = np.arange(4760,4760+num_total_file)
    train_read_index = np.arange(num_total_file)
    if train_only:
        test_read_index = []
    else:
        test_read_index = [i for i in list(range(0,num_total_file,train_test_ratio))]

    train_read_index = [j for j in train_read_index if j not in test_read_index]
    num_batch = len(train_read_index)//batch_size
    num_test_batch = len(test_read_index)//batch_size
    label_test = label[test_read_index]
# 划分测试集 + 打乱数据   后面的模型
else:
    file_nams_epoch = os.listdir(train_dir)[4760:]
    num_total_file = len(file_nams_epoch)
    label = np.load("./label.npy")
    # label_generate = np.load("./label_generate.npy")
    # label = np.concatenate((label,label_generate))
    if train_only:
        test_read_index = []
    else:
        train_read_index = np.arange(4760,4760+num_total_file)
    # train_read_index = np.arange(num_total_file)
    test_read_index = [i for i in list(range(4760,4760+num_total_file, train_test_ratio))]
    # test_read_index = []
    train_read_index = [j for j in train_read_index if j not in test_read_index]
    num_batch = len(train_read_index)//batch_size
    num_test_batch = len(test_read_index)//batch_size
    label_test = label[test_read_index]

#%% 训练2
for epoch in range(epochs_initial+epochs_finetune):
    if epoch >= epochs_initial:
        for layer in mbv2_self_driving.layers[-5:]:
            layer.trainable = True
    # for layer in mbv2_self_driving.layers:
    #     print(layer.trainable)
    np.random.shuffle(train_read_index)
    label_train = label[train_read_index]
    for batch_id in range(num_batch):
        current_batch = extract_one_batch(train_read_index[batch_id*batch_size:(batch_id+1)*batch_size],\
            train_dir)
        trian_loss_metrics =  mbv2_self_driving.train_on_batch(x = current_batch, y = label_train[batch_id*batch_size:(batch_id+1)*batch_size])
        print("%s epoch,%sst batch is training.. "%(epoch+1,batch_id+1))
        print("train loss is ",trian_loss_metrics)
        if batch_id%2 == 20 and batch_id>0:
                for batch_test_id in range(num_test_batch):
                    current_batch = extract_one_batch(test_read_index[batch_test_id*batch_size:(batch_test_id+1)*batch_size],\
                        train_dir)
                    test_loss_metrics = mbv2_self_driving.test_on_batch(current_batch,\
                                    label_test[batch_test_id*batch_size:(batch_test_id+1)*batch_size])
                    # loss = np.mean(np.power(np.sum(np.power((pred - label_test[batch_test_id*batch_size:(batch_test_id+1)*batch_size]),2),axis = 1),0.5))
                    print("test loss and metrics is : ",test_loss_metrics)
    if epoch%1 == 0 and epoch>0 :
        mbv2_self_driving.save("vgg16_"+str(epoch)+".h5")

#%%
