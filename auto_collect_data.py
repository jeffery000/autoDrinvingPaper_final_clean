#/*
# * Copyright 2016-2019. UISEE TECHNOLOGIES LTD. All rights reserved.
# * See LICENSE AGREEMENT file in the project root for full license information.
# */

# This demo code is based on Python3
# If Python2 is needed, please replace usimpy.so in root dirction with lib_x86/lib_py27/usimpy.so

import os
import time
from PIL import Image
import numpy as np
import usimpy
import keras
from keras.models import Model,load_model
import cv2
from keras.applications import VGG16

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# connection
id = usimpy.UsimCreateApiConnection("127.0.0.1", 17771, 5000, 10000)#127.0.0.1

# start simulation
ret = usimpy.UsimStartSim(id, 10000)
print (ret)

## control
#auto = usimpy.UsimManualTransmissionMode()
#auto.brake = False

control = usimpy.UsimSpeedAdaptMode()
control.expected_speed = 0.8 # m/s0.8
control.steering_angle = 0.0 # angle0.0
control.handbrake_on = False 

## action
action = usimpy.UsimVehicleState()
## collision
collision = usimpy.UsimCollision()
## image
image = usimpy.UsimCameraResponse()

count = 0
time1 = 0
time2 = 0
root_dir = './dataset/'
img_dir = root_dir + 'img/'
mkdir(img_dir)
#读取模型
#angl = load_model("vgg16_angle15.h5")
#spee = load_model("vgg16_speed15.h5")
rows = 320
cols = 480
channels = 3
model_vgg = VGG16(include_top=False,input_shape=(rows,cols,channels))
x = model_vgg.get_layer('block3_conv3').output
model_vgg_b3 = Model(input=model_vgg.input, output=x)
model = load_model("log//11_24_keras//KivlNet_part2_16.h5")#9
save_img = False
# 保存路径
date_dir = '../../2019.12.10/auto_save_data/'
img_dir = date_dir + 'img/'
mkdir(img_dir)
button = False
img_text = np.zeros((400,400,3))
cv2.putText(img_text,"press q to stop,press s to start",(10,200),cv2.FONT_HERSHEY_SIMPLEX,0.7,(254,254,254))

while(1):
    # control vehicle via speed & steer
#    print("#####%s"%control.expected_speed)
    ret = usimpy.UsimSetVehicleControlsBySA(id, control)
    # get vehicle post & action
    ret = usimpy.UsimGetVehicleState(id, action)
    # get collision
    ret =usimpy.UsimGetCollisionInformation(id, collision)
    # get RGB image
    ret = usimpy.UsimGetOneCameraResponse(id, 0, image)
    # save image
    img = np.array(image.image[0:480*320*3])
    img = img.reshape((320, 480, 3))
    img_to_save = img.copy()
#    img_PIL = Image.fromarray(np.uint8(img))
    #### pridiction #####
    img_batch = np.expand_dims(img, axis=0)
    conv_img = model_vgg_b3.predict(img_batch)  # conv_img 卷积结果
    conv_img_dst = conv_img[0,:,:,240]
    conv_img_dst_nor = conv_img_dst/conv_img_dst.max()
    conv_img_dst_nor[0:40,:] = 0
#    conv_img_dst_nor = np.array(conv_img_dst_nor,np.uint8)
    conv_img_dst_nor_copy = cv2.resize(conv_img_dst_nor,(240,160))
    # cv2.imshow("feature_map",conv_img_dst_nor_copy)
    # if cv2.waitKey(1) == ord('q'):
    #     break

    # img = img / 255
    img = np.expand_dims(conv_img_dst_nor,axis=0)
    img = np.expand_dims(img,axis=3)
#    pred1 = angl.predict(img, batch_size=1, verbose=0)
#    pred2 = spee.predict(img, batch_size=1, verbose=0)*19
    pred = model.predict(img, batch_size=1, verbose=0)
#    ##### 11.28 #####

#    if abs(pred[0][0]) <= 0.2:
#        pred[0][1]*=1.8
#    elif abs(pred[0][0]) <= 0.5:
#        pred[0][1]*=2.5
#    elif abs(pred[0][0]) <= 1.5:
#        pred[0][1]*=2.5
#    elif abs(pred[0][0]) <= 2.0:
#        pred[0][1]*=2.0
#    else:
#        pred[0][1]*=1.8

# 保存图片和标签
    cv2.imshow('img_text',img_text)
    button_original = cv2.waitKey(1)
    if button_original == ord('q'):
        button = False
        img_text = np.zeros((400,400,3))
        cv2.putText(img_text,"stop record...",(10,200),cv2.FONT_HERSHEY_SIMPLEX,0.7,(254,254,254))

    if button_original == ord('s'):
        button = True
        img_text = np.zeros((400,400,3))
        cv2.putText(img_text,"start record...",(10,200),cv2.FONT_HERSHEY_SIMPLEX,0.7,(254,254,254))

    if button_original == ord('b'):
        break
    # save image
    if button:
        img_PIL = Image.fromarray(np.uint8(img_to_save))
        img_PIL.save(img_dir+str(count)+'.png', 'png')
        # save pose & action
        with open(date_dir + 'action_pose.txt', 'a+') as f:
            f.write(str(count)+',' \
                     +str(action.pose.position.y_val)+','+ str(action.pose.position.z_val)+','+str(action.pose.rotation.x_val)+','+str(action.pose.rotation.y_val)+','\
                     +str(action.pose.rotation.z_val)+','+str(float(pred[0][0]))+','+ str(float(pred[0][1]))+'\n')
            count += 1
# 保存图片和标签

    control.expected_speed = float(pred[0][1]) # m/s0.8
#    control.expected_speed = 10
#    control.expected_speed = float(pred[0][1]) # m/s0.8
 # m/s0.8

    control.steering_angle = float(pred[0][0]) # angle0.0
    
    print("angle is %s,speed is %s"%(str(control.steering_angle),\
                                     str(control.expected_speed)))
    #### pridiction #####
    if save_img:
        img_PIL.save(img_dir+str(action.time_stamp)+'_'+('%010d'%count)+'.png', 'png')
        #img_PIL.save(img_dir+str(action.time_stamp)+'_'+('%010d'%count)+'.png', 'png')
        # save pose & action
        with open(root_dir + 'action_pose.txt', 'a+') as f:
            f.write('%d %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n' % (count, action.time_stamp, action.pose.position.x_val, \
                     action.pose.position.y_val, action.pose.position.z_val, action.pose.rotation.x_val, action.pose.rotation.y_val, \
                     action.pose.rotation.z_val, action.steering_angle, action.forward_speed))
    # auxiliary info
    time1 = action.time_stamp
#    print ('time: %d, steer: %.6f, speed: %.6f, time_gap: %d, collision: %d, collision_time: %d' % (
#            action.time_stamp, action.steering_angle, action.forward_speed, (time1-time2), collision.is_collided, collision.time_stamp))
    time2 = time1
# stop simulation
cv2.destroyAllWindows()
ret = usimpy.UsimStopSim(id)
print (ret)