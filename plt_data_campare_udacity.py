import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import math

if_angle_filter = True
label_dir = "action_pose.txt"
predict_dir = "prediction_udacity_nvNet_5_19.log"
# predict_dir = "prediction_udacity_vgg_fla_pw_3channel_kivinet_5_19.log"
# predict_dir = "prediction_udacity_seg_fla_pw_3channel_kivinet_5_19.log"
# predict_dir = "prediction_udacity_seg_fla_3channel_kivinet_5_19.log"
# predict_dir = "prediction_udacity_inception3_5_19.log"
# predict_dir = "prediction_udacity_vgg_transfer5_19.log"
def angle_filter(angle,filter_size = 11):
    filter = np.ones((1,filter_size),dtype=np.float)/filter_size
    for i in range(filter_size//2,angle.shape[0]-filter_size//2):
        mul = np.matmul(filter,angle[i-filter_size//2:i+filter_size//2+1,:])
        angle[i,:] = mul
    return np.squeeze(angle,axis=1)
def cal_RMSE(predict_np,label_np):
    label_np_cut = label_np[:predict_np.shape[0],:]
    RMSE_angle = math.sqrt(mean_squared_error(predict_np[:,0],label_np_cut[:,0]))
    RMSE_speed = math.sqrt(mean_squared_error(predict_np[:,1],label_np_cut[:,1]))
    return [RMSE_angle,RMSE_speed]
f_label = open(label_dir,'r')
f_label_lines = f_label.readlines()
label = np.zeros((len(f_label_lines),2))
for i,j in enumerate(f_label_lines):
    label[i][0] = float(j.split(",")[-2])
    label[i][1] = float(j.split(",")[-1])
f_label.close()
if if_angle_filter:
    label[:,0] = angle_filter(np.reshape(label[:,0],[-1,1]))
f_predict = open(predict_dir,'r')
f_predict_lines = f_predict.readlines()
angle_predict = []
speed_predict = []
train_data_index = []
test_data_index = []
count_train = 0
count_test = 0
for i in f_predict_lines:
    if i != "":
        angle_predict.append(float(i.split(",")[2]))
        speed_predict.append(float(i.split(",")[3]))
        if (i.split(",")[0]=='train'):
            train_data_index.append(int(i.split(",")[1][:-4]))
            count_train +=1
        else:
            test_data_index.append(int(i.split(",")[1][:-4]))
            count_test +=1
predict_np = np.array([angle_predict,speed_predict]).T
std_predict_np = np.var(predict_np,axis=0)
print(std_predict_np)
RMSE_angle_train,RMSE_speed_train = cal_RMSE(predict_np[:count_train],label[train_data_index])
RMSE_angle_test,RMSE_speed_test = cal_RMSE(predict_np[count_train:],label[test_data_index])
print("Train angle RMSE:%s,Train speed RMSE:%s\n \
    Test angle RMSE:%s,Test speed RMSE:%s"%(\
        RMSE_angle_train,RMSE_speed_train,RMSE_angle_test,RMSE_speed_test))
plt.figure("angle")
plt.subplots_adjust(wspace =0.2, hspace =0.5)
plt.subplot(211)
plt.title("angle train,RMSE:%.2f"%RMSE_angle_train)
plt.plot(train_data_index,label[train_data_index,0],label = "label", color='r', linestyle='--')
plt.plot(train_data_index,np.array(angle_predict[:count_train]),label = "train_predict", color='navy', linestyle='--')
plt.legend()
plt.subplot(212)
plt.title("angle test,RMSE:%.2f"%RMSE_angle_test)
plt.plot(test_data_index,label[test_data_index,0],label = "label", color='r', linestyle='--')
plt.plot(test_data_index,np.array(angle_predict[count_train:count_train+count_test]),label = "test_predict", color='y',linestyle="--")
plt.legend()
plt.savefig("CONV_angle.jpg")

plt.figure("speed")
plt.subplots_adjust(wspace =0.2, hspace =0.5)
plt.subplot(211)
plt.title("speed train,RMSE:%.2f"%RMSE_speed_train)
plt.plot(train_data_index,label[train_data_index,1],label = "label", color='r', linestyle='--')
plt.plot(train_data_index,np.array(speed_predict[:count_train]),label = "train_predict", color='navy', linestyle='--')
plt.legend()
plt.subplot(212)
plt.title("speed test,RMSE:%.2f"%RMSE_speed_test)
plt.plot(test_data_index,label[test_data_index,1],label = "label", color='r', linestyle='--')
plt.plot(test_data_index,np.array(speed_predict[count_train:count_train+count_test]),label = "test_predict", color='y',linestyle="--")
plt.legend()
plt.savefig("CONV_speed.jpg")

plt.show()
