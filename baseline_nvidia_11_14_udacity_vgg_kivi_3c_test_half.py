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
from keras.regularizers import l2

#%% 超参数
batch_size = 64
cols = 160
rows = 80
channels = 6
train_dir = "D:/dataset/selfdriving-car-simulator_Uisee1_3/img"
vgg_cam_dir = "D:/dataset/selfdriving-car-simulator_Uisee1_3/img_vgg_cam"
label_dir = "D:/dataset/selfdriving-car-simulator_Uisee1_3/action_pose.txt"
log_dir="log/5_19ep_udacity_3c_vgg_kivi_2pred_test_half_no_last_pool/"
model_save_basename = "KivlNet_part2_"
epochs_initial = 0
epochs_finetune = 30
train_test_ratio = 5
if_save_model_trained = True
save_interval = 1
train_only = False
train_from_pretrained = False
pretrained_model = "log/4_23ep_udacity_3c_vgg_kivi_2pred_test_half/KivlNet_part2_15.h5"
preWhiten = False
speed_angle_weights_ratio = 1 #影响不大
angle_filter_control = True
img_last_name = ".jpg"
if not os.path.isdir("log/"):
    os.mkdir("log/")
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

#%% 构建my model
model = Sequential()
keras.layers.Activation
model.add(Conv2D(32, (5, 5),input_shape=(rows,cols,channels)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(32, (5, 5)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(BatchNormalization())
# model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(500,activation='relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.7))
model.add(Dense(100,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(20,activation='relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.7))
model.add(Dense(2))
# model.compile(loss=keras.losses.MSE,
#               optimizer=keras.optimizers.SGD(lr=1e-3, decay=1e-4, momentum=0.9, nesterov=True),
#               metrics=['mse'])
model.compile(loss=keras.losses.MSE,
              optimizer=keras.optimizers.Adam(),
              metrics=['mse'])

model.save('my_model.h5')

tensorboard = TensorBoard(log_dir=log_dir)
#%%读取模型
if train_from_pretrained:
    model = load_model(pretrained_model)
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
        img = cv2.imread(train_dir+"/"+str(file_name)+img_last_name)#[:,:,0:1]/255
        img = cv2.resize(img,(cols,rows))/255
        img_vgg = cv2.imread(vgg_cam_dir+"/"+str(file_name)+img_last_name)/255#[:,:,0:1]/255

        # img = np.expand_dims(img, axis=2)
        if preWhiten:
            img = prewhiten(img)
        # img = cv2.resize(img,(shape_img[1],shape_img[0]))
        # img = img/255
        current_batch[index_file_name,:,:,:3] = img
        current_batch[index_file_name,:,:,3:] = img_vgg
    return current_batch
def angle_filter(angle,filter_size = 11):
    filter = np.ones((1,filter_size),dtype=np.float)/filter_size
    for i in range(filter_size//2,angle.shape[0]-filter_size//2):
        mul = np.matmul(filter,angle[i-filter_size//2:i+filter_size//2+1,:])
        angle[i,:] = mul
    return np.squeeze(angle,axis=1)
#%% 训划分测试集 + 打乱数据   前面的模型
file_nams_epoch = os.listdir(train_dir)
num_total_file = len(file_nams_epoch)
# label = np.load("./label.npy")
f_label = open(label_dir,'r')
f_label_lines = f_label.readlines()
label = np.zeros((len(f_label_lines),2))
for i,j in enumerate(f_label_lines):
    label[i][0] = float(j.split(",")[-2])
    label[i][1] = float(j.split(",")[-1])/speed_angle_weights_ratio
f_label.close()
if angle_filter_control:
    label[:,0] = angle_filter(np.reshape(label[:,0],[-1,1]))
train_read_index = np.arange(num_total_file)
if train_only:
    test_read_index = []
else:
    test_read_index = [i for i in list(range(0,num_total_file//train_test_ratio,1))]

train_read_index = [j for j in train_read_index if j not in test_read_index]
num_batch = len(train_read_index)//batch_size
num_test_batch = len(test_read_index)//batch_size
# label = label[:,0]
label_test = label[test_read_index]

#%% 开始训练
f_log_train = open(log_dir+"train_loss.log",'a+')
f_log_test = open(log_dir+"test_loss.log",'a+')
for epoch in range(epochs_initial+epochs_finetune):
    if epoch >= epochs_initial:
        for layer in model.layers[-5:]:
            layer.trainable = True
    # for layer in model.layers:
    #     print(layer.trainable)
    np.random.shuffle(train_read_index)
    label_train = label[train_read_index]
    for batch_id in range(num_batch):
        current_batch = extract_one_batch(train_read_index[batch_id*batch_size:(batch_id+1)*batch_size],\
            train_dir)
        trian_loss_metrics =  model.train_on_batch(x = current_batch, y = label_train[batch_id*batch_size:(batch_id+1)*batch_size])
        print("%s epoch,%sst batch is training.. "%(epoch+1,batch_id+1))
        print("train loss is ",trian_loss_metrics)
        f_log_train.write("epoch:"+str(epoch+1)+" "+"batch:"+str(batch_id+1)+" "+str(trian_loss_metrics[0])+"\n")
        f_log_train.flush()

    for batch_test_id in range(num_test_batch):
        current_batch = extract_one_batch(test_read_index[batch_test_id*batch_size:(batch_test_id+1)*batch_size],\
            train_dir)
        test_loss_metrics = model.test_on_batch(current_batch,\
                        label_test[batch_test_id*batch_size:(batch_test_id+1)*batch_size])
        # loss = np.mean(np.power(np.sum(np.power((pred - label_test[batch_test_id*batch_size:(batch_test_id+1)*batch_size]),2),axis = 1),0.5))
        print("test loss and metrics is : ",test_loss_metrics)
        f_log_test.write("epoch:"+str(epoch+1)+" "+\
            "batch:"+str(batch_test_id+1)+" "+str(test_loss_metrics[0])+"\n")
        f_log_test.flush()
    if if_save_model_trained:
        if (epoch+1)%save_interval == 0 and epoch>0 :
            model.save(log_dir+model_save_basename+str(epoch+1)+".h5")
f_log_train.close()
f_log_test.close()

