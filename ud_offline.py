
import numpy as np
import cv2
import os
import math
original_dir="D:/dataset/selfdriving-car-simulator_Uisee1_3/img/"

label_dir = "action_pose.txt"
predict_dir1 = "prediction_udacity_nvNet_5_19.log"
predict_dir5 = "prediction_udacity_vgg_fla_pw_3channel_kivinet_5_19.log"
predict_dir4 = "prediction_udacity_seg_fla_pw_3channel_kivinet_5_19.log"
predict_dir3 = "prediction_udacity_inception3_5_19.log"
predict_dir2 = "prediction_udacity_vgg_transfer5_19.log"
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
if True:
    label[:,0] = angle_filter(np.reshape(label[:,0],[-1,1]))

base = 8652
index = 1200 #100 1720   1660 150 1000 1200
k=100
f_all = []
f_l = open(label_dir,"r").readlines()[index]
f_p1 = open(predict_dir1,"r").readlines()[base+index]
f_p2 = open(predict_dir2,"r").readlines()[base+index]
f_p3 = open(predict_dir3,"r").readlines()[base+index]
f_p4 = open(predict_dir4,"r").readlines()[base+index]
f_p5 = open(predict_dir5,"r").readlines()[base+index]
f_all.append(f_l)
f_all.append(f_p1)
f_all.append(f_p2)
f_all.append(f_p3)
f_all.append(f_p4)
f_all.append(f_p5)
img = cv2.imread(original_dir+str(index)+".jpg")
color = [(0,0,255),(0,255,0),(255,0,0),(238, 203, 173),(139, 117, 0),( 0, 165,255)]
for id,f in enumerate(f_all):
    if id==0:
        steer = label[index][0]
        speed= label[index][1]
    else:
        steer,speed = f.split(",")[-2:]
        steer = float(steer)
        speed = float(speed)
    if steer>0:
        steer_sqrt = math.sqrt(abs(steer))
    else:
        steer_sqrt = -math.sqrt(abs(steer))
    x = np.array([160,160+0.3*steer_sqrt*k,160+steer_sqrt*k])
    y = np.array([150,100,50])
    poly = np.poly1d(np.polyfit(x, y, 2))
    font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(img, "Steer:%.3f"%steer, (50, 250), font, 0.5, (255, 255, 255), 1)
    # cv2.putText(img, "Speed:15", (50, 270), font, 0.5, (255, 255, 255), 1)
    # cv2.line(img,(320+int(steer_sqrt*k),100),(320+int(steer_sqrt*k),120),(0,255,0),5)
    # cv2.line(img,(320+int(steer_sqrt*k),100),(300+int(steer_sqrt*k),100),(0,255,0),5)
    if int(160+steer_sqrt*k)<160:
        mi = int(160+steer_sqrt*k)
        ma = 160
    else:
        ma = int(160+steer_sqrt*k)
        mi = 160
    for i in range(mi,ma,1):
        y_i = int(poly(i) )
        cv2.circle(img, (i, y_i), 1, color[id], 2, 8, 0)
    cv2.imshow("img",img)
    cv2.waitKey()
cv2.imwrite("ud_poly/"+str(index)+".jpg",img)
# cv2.destroyAllWindows()

# alphaReserve = 0.8
# BChannel = 255
# GChannel = 0
# RChannel = 0
# yMin = 237
# yMax = 277
# xMin = 45
# xMax = 170

# img[yMin:yMax, xMin:xMax, 0] = img[yMin:yMax, xMin:xMax, 0] * alphaReserve + BChannel * (1 - alphaReserve)
# img[yMin:yMax, xMin:xMax, 1] = img[yMin:yMax, xMin:xMax, 1] * alphaReserve + GChannel * (1 - alphaReserve)
# img[yMin:yMax, xMin:xMax, 2] = img[yMin:yMax, xMin:xMax, 2] * alphaReserve + RChannel * (1 - alphaReserve)


# steer = pred[0][0]
# k = 100
# if steer>0:
#     steer_sqrt = math.sqrt(abs(steer))
# else:
#     steer_sqrt = -math.sqrt(abs(steer))
# x = np.array([320,320+0.3*steer_sqrt*k,320+steer_sqrt*k])
# y = np.array([200,150,100])
# poly = np.poly1d(np.polyfit(x, y, 2))
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img, "Steer:%.3f"%steer, (50, 250), font, 0.5, (255, 255, 255), 1)
# cv2.putText(img, "Speed:15", (50, 270), font, 0.5, (255, 255, 255), 1)
# # cv2.line(img,(320+int(steer_sqrt*k),100),(320+int(steer_sqrt*k),120),(0,255,0),5)
# # cv2.line(img,(320+int(steer_sqrt*k),100),(300+int(steer_sqrt*k),100),(0,255,0),5)
# if int(320+steer_sqrt*k)<320:
#     mi = int(320+steer_sqrt*k)
#     ma = 320
# else:
#     ma = int(320+steer_sqrt*k)
#     mi = 320
# for i in range(mi,ma,1):
#     y_i = int(poly(i) )
#     cv2.circle(img, (i, y_i), 3, ( 197, 205,122), 3, 8, 0)
# cv2.imshow("img",img)
# cv2.waitKey()
# cv2.destroyAllWindows()
        