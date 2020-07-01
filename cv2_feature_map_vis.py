#%%
from keras.models import Model
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation
from pylab import *
import keras
from keras.models import load_model
from keras.applications import VGG16
import numpy as np
import os

#%%original vgg16
rows = 320
cols = 480
channels = 3
model = VGG16(include_top=False,input_shape=(rows,cols,channels))
#model = load_model("log/navidia_no_abn/nvidia_model_angle_spped_12.h5")
#x = model.get_layer('conv2d_4').output
x = model.get_layer('block3_conv3').output
model = Model(input=model.input, output=x)

#%%
img_shown_dir = "C:/Users/SHJ/Desktop/sim/total_no_abnormal_original_11_16/img/"
img_names = os.listdir(img_shown_dir)
for img_name in img_names:
    img = cv2.imread(img_shown_dir+img_name)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(cols,rows))
    img_batch = np.expand_dims(img, axis=0)
    conv_img = model.predict(img_batch)  # conv_img 卷积结果
    conv_img_dst = conv_img[0,:,:,240]
    conv_img_dst_nor = conv_img_dst/conv_img_dst.max()*255
    conv_img_dst_nor[:40,:] = 0
    conv_img_dst_nor = np.array(conv_img_dst_nor,np.uint8)
#    #hough
#    
#    #conv_img_dst_nor = cv2.GaussianBlur(conv_img_dst_nor,(3,3),0)
#    edges = cv2.Canny(conv_img_dst_nor,50,150,apertureSize = 3)
##    edges = cv2.Sobel(edges,cv2.CV_64F,1,0)    
##    edges = np.uint8(np.absolute(edges))
##    lines = cv2.HoughLines(edges,2,np.pi/180,100)
#    #lines = cv2.HoughLines(edges,1,np.pi/180,50)
#    
#    
#    minLineLength = 100
#    maxLineGap = 15
#    lines = cv2.HoughLinesP(edges,1,np.pi/180,80,minLineLength,maxLineGap)
#
#    
##    if not lines is None:
##        for line in lines:
##            rho,theta = line[0]
##            a = np.cos(theta)
##            b = np.sin(theta)
##            x0 = a*rho
##            y0 = b*rho
##            x1 = int(x0 + 1000*(-b))
##            y1 = int(y0 + 1000*(a))
##            x2 = int(x0 - 1000*(-b))
##            y2 = int(y0 - 1000*(a))
##            cv2.line(conv_img_dst_nor,(x1,y1),(x2,y2),(255),1)
#    if not lines is None:
#        for x1,y1,x2,y2 in lines[0]:
#            cv2.line(conv_img_dst_nor,(x1,y1),(x2,y2),(255),1)
#
#    #hough
    conv_img_dst_nor = cv2.resize(conv_img_dst_nor,(cols*2,rows*2))#conv_img_dst_nor
    cv2.imshow("img",conv_img_dst_nor)
    if cv2.waitKey(100) == ord('q'):
        break
cv2.destroyAllWindows()
#conv_img_sum = np.squeeze(conv_img_sum,axis=0)
#conv_img_sum = 1/(conv_img_sum/conv_img_sum.max())
#ret,conv_img_sum=cv2.threshold(np.expand_dims(conv_img_sum,axis=2),8,1,cv2.THRESH_BINARY)
#conv_img_sum[:45,:] = 0
# cv2.imwrite("1573518191670290_0000005753.jpg",conv_img_sum*255)
#conv_img_sum[conv_img_sum>12] =100
#%%
plt.imshow(conv_img_sum,cmap='Greys')#='Greys')
plt.show()
#%%
visualize_feature_map(conv_img)



# %%
