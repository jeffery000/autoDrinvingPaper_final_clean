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
# plt.figure(figsize=(10,15))
fig,axs = plt.subplots(2, 1)
fig.set_figheight(15)
fig.set_figwidth(10)
ax1 = axs[0]
ax1.set_title("train",fontsize=16,color='b')
ax1.set_xlabel("batchs")
ax1.set_ylabel("mse")
for sub_dir in log_sub_dir:
    f_train = open(log_dir+sub_dir+"/"+train_loss_name,'r')
    loss_train = []
    for l in f_train.readlines():
        loss_train.append(float(l.split()[-1]))
    ax1.plot(range(len(loss_train)),loss_train)
ax1.legend(labels=log_sub_dir,loc='best')
ax2 = axs[1]
ax2.set_title("test",fontsize=16,color='b')
ax2.set_xlabel("epochs")
ax2.set_ylabel("mse")
for sub_dir in log_sub_dir:
    f_test = open(log_dir+sub_dir+"/"+test_loss_name,'r')
    loss_test = []
    loss_test_mean = []
    for l in f_test.readlines():
        loss_test.append(float(l.split()[-1]))
    for i in range(50):
        loss_test_mean.append(sum(loss_test[i * 63:(i+1) * 63]) / 63)
    ax2.plot(range(len(loss_test_mean)), loss_test_mean)
ax2.legend(labels=log_sub_dir,loc='best')
plt.show()

