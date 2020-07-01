import cv2
import numpy as np 
from matplotlib import pyplot as plt
import seaborn as sns
from pylab import *
plt.rc('font',family='Times New Roman') 
plt.rc('font',size=24) 
def normlization(batch):
    if False:
        print("ndims is not 2!")
    else:
        num_max = np.max(batch)
        num_min = np.min(batch)
        batch = (batch-num_min)/(num_max-num_min)
    return batch
conv1 = cv2.imread("conv1_sum.jpg")
conv2 = cv2.imread("conv2_sum.jpg")
conv3 = cv2.imread("conv3_sum.jpg")
conv4 = cv2.imread("conv4_sum.jpg")
conv1 = cv2.cvtColor(conv1,cv2.COLOR_BGR2GRAY)
conv2 = cv2.cvtColor(conv2,cv2.COLOR_BGR2GRAY)
conv3 = cv2.cvtColor(conv3,cv2.COLOR_BGR2GRAY)
conv4 = cv2.cvtColor(conv4,cv2.COLOR_BGR2GRAY)
conv1 = cv2.resize(conv1,(160,80))
conv2 = cv2.resize(conv4,(80,40))
conv3 = cv2.resize(conv3,(40,20))
conv4 = cv2.resize(conv4,(40,20))

img_ori = cv2.imread(r'E:\dateset\selfdriving-car-simulator_Uisee1_3\img\556.jpg')
img_ori = cv2.resize(img_ori,(160,80))


conv4 = normlization(conv4)
conv3 = normlization(conv3)
con = (conv3*conv4)
con = cv2.resize(con,(80,40),interpolation=2)
conv2 = normlization(conv2)
con = normlization(con)
con = (conv2*con)
con = cv2.resize(con,(160,80),interpolation=2)
con = normlization(con)
conv1 = normlization(conv1)
con = (con*conv1)
con = normlization(con)
con_write = (1-con)*255
con_write1 = np.power(con_write,10)
con_write1 = normlization(con_write1)
con_write2 = np.power(con_write,5)
con_write2 = normlization(con_write2)
# thresh = 0.3
# for i in range(80):
#     for j in range(160):
#         if con_write1[i,j] > thresh:
#             img_ori[i,j] = [0,255-(con_write1[i,j]-0.3)*255,255]
con_write1[con_write1<0.3] = 0
con_write1 = np.uint8(255 * con_write1)
im_color = cv2.applyColorMap(con_write1, cv2.COLORMAP_JET)
img_ori = img_ori+im_color*0.3

cv2.imwrite("backvis.jpg",img_ori)
plt.imshow(con_write2)
axis('off')
plt.colorbar(orientation='horizontal')

plt.savefig("backvis.png",dpi=500,bbox_inches = 'tight')
# plt.show()
#,interpolation=CV_INTER_LINEAR