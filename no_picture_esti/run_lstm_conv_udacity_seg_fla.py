"""
import library
"""
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
train_from_begin = False
label_dir = "action_pose.txt"
model_LSTM_name = "lstm_model_udacity.h5"
model_CONV_name = "../log/5_19ep_udacity_3c_seg_fla_pw_2pred_test_half/KivlNet_part2_30.h5"
test_dir = "D:/dataset/selfdriving-car-simulator_Uisee1_3/img"
vgg_cam_dir = "D:/dataset/selfdriving-car-simulator_Uisee1_3/img_seg_cam"
look_back = 6
train_test_ratio = 5
predict_dir = "prediction_ud_seg_fla_5_19.log"
cols = 160
rows = 80
channels = 6
preWhiten = False

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
def angle_filter(angle,filter_size = 11):
    filter = np.ones((1,filter_size),dtype=np.float)/filter_size
    for i in range(filter_size//2,angle.shape[0]-filter_size//2):
        mul = np.matmul(filter,angle[i-filter_size//2:i+filter_size//2+1,:])
        angle[i,:] = mul
    return np.squeeze(angle,axis=1)

f_label = open(label_dir,'r')
f_label_lines = f_label.readlines()
label = np.zeros((len(f_label_lines),2))
for i,j in enumerate(f_label_lines):
    label[i][0] = float(j.split(",")[-2])
    label[i][1] = float(j.split(",")[-1])
f_label.close()
label[:,0] = angle_filter(np.reshape(label[:,0],[-1,1]))
dataset = label
def create_dataset(dataset, look_back):
	dataX = np.zeros((len(dataset)-look_back-1,look_back,2))
	dataY = np.zeros((len(dataset)-look_back-1,2))
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back),:]
		dataX[i] = a
		b = dataset[i + look_back,:]
		dataY[i] = b
	return dataX.swapaxes(1,2), dataY
dataset_all_X, dataset_all_Y = create_dataset(dataset, look_back)
train_read_index = list(range(dataset_all_X.shape[0]))
test_read_index = [i for i in list(range(6,dataset_all_X.shape[0]//train_test_ratio,1))]
train_read_index = [j for j in train_read_index if j not in test_read_index]
trainX = dataset_all_X[train_read_index,:,:]
trainY = dataset_all_Y[train_read_index,:]
testX = dataset_all_X[test_read_index,:,:]
testY = dataset_all_Y[test_read_index,:]
# train_Y_original = scaler.inverse_transform(trainY)
# test_Y_original = scaler.inverse_transform(testY)
def extract_one_batch(list_train_index, train_dir, shape_img = [rows,cols,channels]):
    current_batch = np.zeros((len(list_train_index), shape_img[0], shape_img[1],\
        shape_img[2]))
    for index_file_name,file_name in enumerate(list_train_index):
        img = cv2.imread(train_dir+"/"+file_name)
        img = cv2.resize(img,(cols,rows))/255
        img_vgg = cv2.imread(vgg_cam_dir+"/"+file_name)
        img_vgg = cv2.resize(img,(cols,rows))/255#[:,:,0:1]/255

        # img = np.expand_dims(img, axis=2)
        if preWhiten:
            img = prewhiten(img)
        # img = cv2.resize(img,(shape_img[1],shape_img[0]))
        # img = img/255
        current_batch[index_file_name,:,:,:3] = img
        current_batch[index_file_name,:,:,3:] = img_vgg

    return current_batch
model_LSTM = load_model(model_LSTM_name)
model_CONV = load_model(model_CONV_name)
num_batch = len(test_read_index)


file_nams_epoch = os.listdir(test_dir)
file_nams_epoch.sort(key = lambda x: int(x[:-4]))
f_prediction = open(predict_dir,"w")
predict_list = []
for batch_id in range(num_batch-2):
    file_name_current = []
    for i in range(look_back+1):
        file_name_current.append(str(test_read_index[batch_id+2]-look_back+i)+".jpg")
    current_batch = extract_one_batch(file_name_current,test_dir)
    outout_conv1 = model_CONV.predict(current_batch[:-1,:,:,:])
    outout_conv2 = model_CONV.predict(current_batch[-1:,:,:,:])
    # outout_conv1_input = np.expand_dims(outout_conv1.swapaxes(0,1),axis=0)
    scaler = MinMaxScaler().fit(outout_conv1)
    outout_conv1_input = scaler.transform(outout_conv1)
    outout_conv1_input = np.expand_dims(outout_conv1_input.swapaxes(0,1),axis=0)
    outout_lstm = model_LSTM.predict(outout_conv1_input)
    outout_lstm = scaler.inverse_transform(outout_lstm)
    angle = (1.2*outout_conv2[0,0]+0.8*outout_lstm[0,0])/2
    speed = (0.8*outout_conv2[0,1]+1.2*outout_lstm[0,1])/2
    predict_list.append([angle,speed])
    f_prediction.write(str(angle)+","+str(speed)+"\n")
f_prediction.close()
# def cal_RMSE(predict_np,label_np):
#     label_np_cut = label_np[:predict_np.shape[0],:]
#     RMSE_angle = math.sqrt(mean_squared_error(predict_np[:,0],label_np_cut[:,0]))
#     RMSE_speed = math.sqrt(mean_squared_error(predict_np[:,1],label_np_cut[:,1]))
#     return [RMSE_angle,RMSE_speed]
# RMSE_angle_test,RMSE_speed_test = cal_RMSE(np.array(predict_list),np.array(test_Y_original))
# print("Test angle RMSE:%s,Test speed RMSE:%s"%(RMSE_angle_test,RMSE_speed_test))

# plt.figure("CONV_LSTM_angle")
# plt.title("angle test,RMSE:%.2f"%RMSE_angle_test)
# plt.plot(test_read_index,testY[:,0],label = "label", color='r', linestyle='--')
# plt.plot(test_read_index,np.array(predict_list)[:,0],label = "test_predict", color='y',linestyle="--")
# plt.legend()
# plt.savefig("CONV_LSTM_angle.jpg")

# plt.figure("CONV_LSTM_speed")
# plt.title("speed test,RMSE:%.2f"%RMSE_speed_test)
# plt.plot(test_read_index,testY[:,1],label = "label", color='r', linestyle='--')
# plt.plot(test_read_index,np.array(predict_list)[:,1],label = "test_predict", color='y',linestyle="--")
# plt.legend()
# plt.savefig("CONV_LSTM_speed.jpg")
# plt.show()