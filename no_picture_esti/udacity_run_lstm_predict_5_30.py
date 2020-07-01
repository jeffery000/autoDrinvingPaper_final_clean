"""
import library
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import TensorBoard

# fix random seed for reproducibility
train_from_begin = False
label_dir = "../prediction_udacity_seg_fla_pw_3channel_kivinet_5_19.log"
f_w_name = "udacity_run_lstm_predict_seg_pw.log"
epochs = 300
model_name = "lstm_model_udacity.h5"
look_back = 6
train_test_ratio = 5
data_cut = False	
data_cut_num = 4350
def angle_filter(angle,filter_size = 11):
    filter = np.ones((1,filter_size),dtype=np.float)/filter_size
    for i in range(filter_size//2,angle.shape[0]-filter_size//2):
        mul = np.matmul(filter,angle[i-filter_size//2:i+filter_size//2+1,:])
        angle[i,:] = mul
    return np.squeeze(angle,axis=1)

f_label = open(label_dir,'r')
f_label_lines = f_label.readlines()
f_label_lines_test = []
for i in f_label_lines:
    if i[:4] == "test":
        f_label_lines_test.append(i)
if data_cut:
	f_label_lines = f_label_lines[:data_cut_num]
label = np.zeros((len(f_label_lines_test),2))
for i,j in enumerate(f_label_lines_test):
    label[i][0] = float(j.split(",")[-2])
    label[i][1] = float(j.split(",")[-1])
f_label.close()
# label[:,0] = angle_filter(np.reshape(label[:,0],[-1,1]))
dataset = label

# split into train and test sets
# train_size = int(dataset.shape[0] * 0.8)
# test_size = dataset.shape[0] - train_size
# train, test = dataset[0:train_size,:], dataset[train_size:dataset.shape[0],:]
# print ("train_data_size: "+str(len(train)), " test_data_size: "+str(len(test)))

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back):
	dataX = np.zeros((len(dataset)-look_back-1,look_back,2))
	dataY = np.zeros((len(dataset)-look_back-1,2))
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back),:]
		dataX[i] = a
		b = dataset[i + look_back,:]
		dataY[i] = b
	return dataX.swapaxes(1,2), dataY


# reshape into X=t and Y=t+1
scaler = MinMaxScaler().fit(dataset)
dataset_scalar = scaler.transform(dataset)
dataset_all_X, dataset_all_Y = create_dataset(dataset_scalar, look_back)
train_read_index = list(range(dataset_all_X.shape[0]))
test_read_index = [i for i in list(range(0,dataset_all_X.shape[0],train_test_ratio))]
train_read_index = [j for j in train_read_index if j not in test_read_index]
trainX = dataset_all_X[train_read_index,:,:]
trainY = dataset_all_Y[train_read_index,:]
testX = dataset_all_X[test_read_index,:,:]
testY = dataset_all_Y[test_read_index,:]
train_Y_original = scaler.inverse_transform(trainY)
test_Y_original = scaler.inverse_transform(testY)
# trainX, trainY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)
print(trainX,trainY)
# reshape input to be [samples, time steps, features]
# trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# print(trainX)
""" The network has a visible layer with 1 input, a hidden layer with
4 LSTM blocks or neurons and an output layer that makes a single value
prediction. The default sigmoid activation function is used for the
LSTM blocks. The network is trained for 100 epochs and a batch size of
1 is used."""

if train_from_begin:
	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(4, input_dim=look_back))
	model.add(Dense(2))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(trainX, trainY, nb_epoch=epochs, batch_size=64, validation_data = (testX,testY),verbose=2,callbacks=[TensorBoard(log_dir='run_comma')])
	model.save(model_name)
else:
	model = load_model(model_name)
# make predictions
Predict = model.predict(dataset_all_X)
Predict = scaler.inverse_transform(Predict)
f_w = open(f_w_name,"w")
for i in range(len(Predict)):
    f_w.write("test,"+str(Predict[i][0])+","+str(Predict[i][1])+"\n")
f_w.close()