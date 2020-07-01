#%%
import matplotlib
from matplotlib import pyplot as plt
import os
#%%
log_dir = "log/"
train_loss_name = "train_loss.log"
test_loss_name = "test_loss.log"
#%%
log_sub_dir = os.listdir(log_dir)
#%%
def cal_batchs(lines_list):
    count = 0 
    for line in lines_list:
        if line[:7] == "epoch:1":
            count+=1
        else:
            break
    return count
def init__plot(fig_title,xlabel,ylabel):
    # plt.grid(axis = "y")
    plt.title(fig_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yticks(range(0,100,10))
def get_data_list(f_name):
    f = open(f_name,'r')
    f_lines = f.readlines()
    f_batch_num = cal_batchs(f_lines)
    loss__batchs = []
    loss__epochs = []
    for l in f_lines:
        loss__batchs.append(float(l.split()[-1]))
    for i in range(len(f_lines)//f_batch_num):
        loss__epochs.append(\
            sum(loss__batchs[i*f_batch_num:(i+1)*f_batch_num])/f_batch_num)
    return [loss__batchs,loss__epochs]
for log in log_sub_dir:
    if log[-4:] == ".jpg":
        continue
    loss_train_batchs,loss_train_epochs = get_data_list(log_dir+log+"/"+train_loss_name)
    loss_test_batchs,loss_test_epochs = get_data_list(log_dir+log+"/"+test_loss_name)

    plt.figure(log)
    plt.subplots_adjust(wspace =0.2, hspace =0.5)

    plt.subplot(2,2,1)
    init__plot("trainLoss_batch","batchs","mse")
    plt.plot(range(len(loss_train_batchs)),loss_train_batchs)    
    
    plt.subplot(2,2,2)
    init__plot("trainLoss_epoch","epochs","mse")
    plt.plot(range(len(loss_train_epochs)),loss_train_epochs)
    
    plt.subplot(2,2,3)
    init__plot("testLoss_batch","batchs","mse")
    plt.plot(range(len(loss_test_batchs)),loss_test_batchs)
    
    plt.subplot(2,2,4)
    init__plot("testLoss_epoch","epochs","mse")
    plt.plot(range(len(loss_test_epochs)),loss_test_epochs)

    plt.savefig(log_dir+log+".jpg")
plt.show()

