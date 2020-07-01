#/*
# * Copyright 2016-2019. UISEE TECHNOLOGIES LTD. All rights reserved.
# * See LICENSE AGREEMENT file in the project root for full license information.
# */

# This test code is based on Python3
# PYTHON DEPENDENCIES:
# - tensorflow-gpu==1.13.1
# - keras==2.3.1
# - numpy==1.17.0
# RUN TURORIALS:
# Put uisee666.py and two separate training model files in the same directory, 
# and run the command through CMD：
#   `python3 Uisee666.py`

# Author: hnu-Kivl

import os
import time
from PIL import Image
import numpy as np
import usimpy
import keras
from keras.models import Model,load_model

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# connection
id = usimpy.UsimCreateApiConnection("127.0.0.1", 17771, 5000, 10000)

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
#root_dir = './dataset/'
#img_dir = root_dir + 'img/'
#mkdir(img_dir)
#读取模型

model_kivl_part1 = load_model("KivlNet_part1.h5")
model_kivl_part2 = load_model("KivlNet_part2.h5")
save_img = False
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

    #************ pridiction begin***************
    img_batch = np.expand_dims(img, axis=0)
    conv_img = model_kivl_part1.predict(img_batch)  
    conv_img_dst = conv_img[0,:,:,240]
    conv_img_dst_norm = conv_img_dst/conv_img_dst.max()
    conv_img_dst_norm[0:40,:] = 0
    img = np.expand_dims(conv_img_dst_norm,axis=0)
    img = np.expand_dims(img,axis=3)
    pred = model_kivl_part2.predict(img, batch_size=1, verbose=0)
    control.expected_speed = float(pred[0][1]) # speed control

    control.steering_angle = float(pred[0][0]) # angle control
    
    print("angle is %s,speed is %s"%(str(control.steering_angle),\
                                     str(control.expected_speed)))
    #************ pridiction end***************

    # set save_img as False
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
    count = count + 1
# stop simulation
ret = usimpy.UsimStopSim(id)
print (ret)