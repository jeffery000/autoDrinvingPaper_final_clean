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
model = load_model("log/full_no_dropout/nvidia_model_angle_spped_56.h5")
save_img = False
def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y
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
#    img_PIL = Image.fromarray(np.uint8(img))
    #### pridiction #####
    img_batch = np.expand_dims(img, axis=0)
    conv_img = model_vgg_b3.predict(img_batch)  # conv_img 卷积结果
    conv_img_sum = np.sum(conv_img,axis=3)
    conv_img_sum = np.squeeze(conv_img_sum,axis=0)
    conv_img_sum = 1/(conv_img_sum/conv_img_sum.max())
    ret,conv_img_sum=cv2.threshold(np.expand_dims(conv_img_sum,axis=2),8,1,cv2.THRESH_BINARY)
    conv_img_sum[:40,:] = 0

    # img = img / 255
    img = np.expand_dims(conv_img_sum,axis=0)
    img = np.expand_dims(img,axis=3)
#    pred1 = angl.predict(img, batch_size=1, verbose=0)
#    pred2 = spee.predict(img, batch_size=1, verbose=0)*19
    pred = model.predict(img, batch_size=1, verbose=0)
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
    count = count + 1
# stop simulation
ret = usimpy.UsimStopSim(id)
print (ret)