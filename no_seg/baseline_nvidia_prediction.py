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
from keras.callbacks import TensorBoard
from keras import Sequential

#%% 超参数
batch_size = 32
cols = 80
rows = 40
channels = 3
# train_dir = "/home/stephan/Yushi/dataset/img_seg"
# label_dir = "/home/stephan/Yushi/dataset/action_pose.txt"
test_dir = "E:/dateset/selfdriving-car-simulator_Uisee_ds1_5/img"
train_test_ratio = 10
preWhiten = False
model_name = "log/3_20_50ep_udac_no_seg/KivlNet_part2_50.h5"
prediction_dir = "prediction.log"
#%% 构建my model
model = load_model(model_name)
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
        img = cv2.imread(train_dir+"/"+file_name)
        img = cv2.resize(img,(80,40))/255
        if preWhiten:
            img = prewhiten(img)
        # img = cv2.resize(img,(shape_img[1],shape_img[0]))
        # img = img/255
        current_batch[index_file_name] = img
    return current_batch
#%% 
file_nams_epoch = os.listdir(test_dir)
file_nams_epoch.sort(key = lambda x: int(x[:-4]))
num_total_file = len(file_nams_epoch)

train_read_index = list(range(num_total_file))
test_read_index = [i for i in list(range(0,num_total_file,train_test_ratio))]
train_read_index = [j for j in train_read_index if j not in test_read_index]
test_file_names = []
for ii in test_read_index:
    test_file_names.append(file_nams_epoch[ii])
train_file_names = []
for jj in train_read_index:
    train_file_names.append(file_nams_epoch[jj])



def prediction(num_batch,num_left,file_nams_epoch,f_prediction,mode,model = model,test_dir = test_dir):
    for batch_test_id in range(num_batch):
        current_batch = extract_one_batch(file_nams_epoch[batch_test_id*batch_size:(batch_test_id+1)*batch_size],\
            test_dir)
        outout = model.predict(current_batch)
        for i in range(batch_size):
            f_prediction.write(mode+","+file_nams_epoch[\
                (batch_test_id)*batch_size+i]+","+str(outout[i][0])+","+str(outout[i][1])+"\n")
    if num_left !=0 :
        current_batch = extract_one_batch(file_nams_epoch[(batch_test_id+1)*batch_size:(batch_test_id+1)*batch_size+num_left],\
        test_dir)
        outout = model.predict(current_batch)
        for i in range(num_left):
            f_prediction.write(mode+","+file_nams_epoch[\
                (batch_test_id+1)*batch_size+i]+","+str(outout[i][0])+","+str(outout[i][1])+"\n")

f_prediction = open(prediction_dir,'w')
# preidct  data
num_batch_train = len(train_file_names)//batch_size
num_left_train = len(train_file_names)%batch_size
prediction(num_batch_train,num_left_train,train_file_names,f_prediction,mode = "train")

num_batch_test = len(test_file_names)//batch_size
num_left_test = len(test_file_names)%batch_size
prediction(num_batch_test,num_left_test,test_file_names,f_prediction,mode = "test")

f_prediction.close()
