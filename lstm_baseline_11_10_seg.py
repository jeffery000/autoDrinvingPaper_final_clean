#%%
import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Activation,LSTM,ConvLSTM2D,Conv3D
from keras.layers import AveragePooling2D, Input, Flatten,Dropout,MaxPooling2D
from keras import backend as K
from keras.models import Model
import numpy as np
import os
from keras.applications import MobileNetV2
import cv2
from keras.models import load_model
from keras.callbacks import TensorBoard
#%%
batch_size = 32
cols = 80
rows = 45
channels = 1
sequential_num = 3
train_dir = r"D:\dataset\yushikeji\train_seg"
epochs = 10
train_test_ratio = 5
save_interval = 2
train_only = False

#%%
model_convlstm = Sequential()
model_convlstm.add(ConvLSTM2D(filters=64, kernel_size=(5, 5),
                   input_shape=(sequential_num, rows, cols, channels),
                   padding='same', return_sequences=True))
model_convlstm.add(BatchNormalization())

model_convlstm.add(ConvLSTM2D(filters=96, kernel_size=(5, 5),
                   padding='same', return_sequences=True))
model_convlstm.add(BatchNormalization())

model_convlstm.add(ConvLSTM2D(filters=128, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
model_convlstm.add(BatchNormalization())

model_convlstm.add(ConvLSTM2D(filters=180, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
model_convlstm.add(BatchNormalization())

# model_convlstm.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
#                activation='sigmoid',
#                padding='same', data_format='channels_last'))
model_convlstm.add(Flatten())
model_convlstm.add(Dense(1))
model_convlstm.compile(loss=keras.losses.mse, optimizer=keras.optimizers.Adam(),metrics=["mae"])
model_convlstm.save("model_convlstm_speed.h5")
#%%
def extract_one_batch(list_train_index, train_dir,sequential_num, shape_img = [rows,cols,channels]):
    current_batch = np.zeros((len(list_train_index), sequential_num, shape_img[0], shape_img[1]))
    for index_file_name,file_name in enumerate(list_train_index):
        for sequential_img in range(sequential_num):
            img = cv2.imread(train_dir+"/"+str(file_name+sequential_num)+".tiff")
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret,img = cv2.threshold(img,10,255,cv2.THRESH_BINARY)
            # if preWhiten:
            #     img = prewhiten(img)
            img = cv2.resize(img,(shape_img[1],shape_img[0]))
            # img = img/255
            current_batch[index_file_name][sequential_img] = img
    current_batch = np.expand_dims(current_batch,axis = 4)
    return current_batch
#%%
file_nams_epoch = os.listdir(train_dir)[:4760][:-sequential_num]
num_total_file = len(file_nams_epoch)
label = np.load("./label.npy")[:,1]
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
#%%
for epoch in range(epochs):
    # 如果用LSTM不能打乱顺序！
    # np.random.shuffle(train_read_index)
    label_train = label[train_read_index]
    for batch_id in range(num_batch):
        current_batch = extract_one_batch(train_read_index[batch_id*batch_size:(batch_id+1)*batch_size],\
            train_dir,sequential_num)
        trian_loss_metrics =  model_convlstm.train_on_batch(x = current_batch, y = label_train[batch_id*batch_size:(batch_id+1)*batch_size])
        print("%s epoch,%sst batch is training.. "%(epoch+1,batch_id+1))
        print("train loss is ",trian_loss_metrics)
        if batch_id%2 == 20 and batch_id>0:
                for batch_test_id in range(num_test_batch):
                    current_batch = extract_one_batch(test_read_index[batch_test_id*batch_size:(batch_test_id+1)*batch_size],\
                        train_dir)
                    test_loss_metrics = model_convlstm.test_on_batch(current_batch,\
                                    label_test[batch_test_id*batch_size:(batch_test_id+1)*batch_size])
                    # loss = np.mean(np.power(np.sum(np.power((pred - label_test[batch_test_id*batch_size:(batch_test_id+1)*batch_size]),2),axis = 1),0.5))
                    print("test loss and metrics is : ",test_loss_metrics)
    if epoch%20 == 0 and epoch>0 :
        mbv2_self_driving.save("model_convlstm_"+str(epoch)+".h5")


# %%
