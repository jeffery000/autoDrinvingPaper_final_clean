import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import math
import random

if_angle_filter = True
if_label_random = False
label_dir = "action_pose.txt"
predict_dir_lstm = "udacity_run_lstm_predict_seg_pw.log"
predict_dir_conv = "../prediction_udacity_seg_fla_pw_3channel_kivinet_5_19.log"
predict_dir_conv_lstm = "prediction_udacity_seg_fla_pw_lstm_conv_no_model_5_30.log"
# predict_dir_lstm = "comma_run_lstm_predict_seg_pw.log"
# predict_dir_conv = "../prediction_comma_seg_fla_pw_3channel_kivinet_5_19.log"

# predict_dir = "prediction_comma_5_19.log"
# predict_dir = "prediction_comma_nvnet_5_19.log"
# predict_dir = "prediction_comma_vgg_transfer_5_19.log"
# predict_dir = "prediction_comma_inception_5_19.log"
# predict_dir = "prediction_comma_seg_fla_5_19.log"
train_test_ratio = 5
def angle_filter(angle,filter_size = 11):
    filter = np.ones((1,filter_size),dtype=np.float)/filter_size
    for i in range(filter_size//2,angle.shape[0]-filter_size//2):
        mul = np.matmul(filter,angle[i-filter_size//2:i+filter_size//2+1,:])
        angle[i,:] = mul
    return np.squeeze(angle,axis=1)
def cal_RMSE(predict_np,label_np):
    # label_np_cut = label_np[7:,:]
    RMSE_angle = math.sqrt(mean_squared_error(predict_np[:,0],label_np[:,0]))
    RMSE_speed = math.sqrt(mean_squared_error(predict_np[:,1],label_np[:,1]))
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
if if_label_random:
    label[:,0] = label[:,0]#+np.random.randn(len(label))/10
    label[:,1] = label[:,1]+np.random.randn(len(label))/3
f_predict_lstm = open(predict_dir_lstm,'r')
f_predict_lines_lstm = f_predict_lstm.readlines()
f_predict_conv = open(predict_dir_conv,'r')
f_predict_lines_conv = f_predict_conv.readlines()
angle_predict = []
speed_predict = []
angle_predict_conv = []
speed_predict_conv = []
train_data_index = []
test_data_index = []
count_train = 0
count_test = 0
test_read_index = [i for i in list(range(0,label.shape[0]//train_test_ratio,1))]
label = label[test_read_index[0:]]
label = label[7:]
for i in f_predict_lines_lstm:
    if i != "":
        angle_predict.append(float(i.split(",")[-2]))
        speed_predict.append(float(i.split(",")[-1]))
for i in f_predict_lines_conv:
    if i[:4] == "test":
        angle_predict_conv.append(float(i.split(",")[-2]))
        speed_predict_conv.append(float(i.split(",")[-1]))
predict_np = np.array([angle_predict,speed_predict]).T
predict_np_conv = np.array([angle_predict_conv,speed_predict_conv]).T
predict_np_conv = predict_np_conv[7:]
predict_merge = np.zeros([len(predict_np),2])
predict_merge[:,0] = 0.6*predict_np_conv[:,0]+0.4*predict_np[:,0]
predict_merge[:,1] = 0.4*predict_np_conv[:,1]+0.6*predict_np[:,1]
f_no_model = open(predict_dir_conv_lstm,'w')
for i in predict_merge:
    f_no_model.write(str(i[0])+","+str(i[1])+"\n")
f_no_model.close()
predict_merge_var = np.var(predict_merge,axis=0)
print(predict_merge_var)
# RMSE_angle_train,RMSE_speed_train = cal_RMSE(predict_np[:count_train],label[train_data_index])
RMSE_angle_test,RMSE_speed_test = cal_RMSE(predict_merge,label)
# print("Train angle RMSE:%s,Train speed RMSE:%s\n \
#     Test angle RMSE:%s,Test speed RMSE:%s"%(\
#         RMSE_angle_train,RMSE_speed_train,RMSE_angle_test,RMSE_speed_test))
plt.figure("angle")
plt.subplots_adjust(wspace =0.2, hspace =0.5)
plt.subplot(211)
# plt.title("angle train,RMSE:%.2f"%RMSE_angle_train)
# plt.plot(train_data_index,label[train_data_index,0],label = "label", color='r', linestyle='--')
# plt.plot(train_data_index,np.array(angle_predict[:count_train]),label = "train_predict", color='navy', linestyle='--')
# plt.legend()
plt.subplot(212)
plt.title("angle test,RMSE:%.2f"%RMSE_angle_test)
plt.plot(range(len(label)),label[:,0],label = "label", color='r', linestyle='--')
plt.plot(range(len(predict_np)),predict_np[:,0],label = "test_predict", color='y',linestyle="--")
plt.legend()
plt.savefig("CONV_angle.jpg")

plt.figure("speed")
plt.subplots_adjust(wspace =0.2, hspace =0.5)
plt.subplot(211)
# plt.title("speed train,RMSE:%.2f"%RMSE_speed_train)
# plt.plot(train_data_index,label[train_data_index,1],label = "label", color='r', linestyle='--')
# plt.plot(train_data_index,np.array(speed_predict[:count_train]),label = "train_predict", color='navy', linestyle='--')
# plt.legend()
plt.subplot(212)
plt.title("speed test,RMSE:%.2f"%RMSE_speed_test)
plt.plot(range(len(label)),label[:,1],label = "label", color='r', linestyle='--')
plt.plot(range(len(predict_np)),predict_np[:,1],label = "test_predict", color='y',linestyle="--")
plt.legend()
plt.savefig("CONV_speed.jpg")

plt.show()
