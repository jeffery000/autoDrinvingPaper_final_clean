# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:55:50 2019

@author: SHJ
"""

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
import cv2

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# connection
id = usimpy.UsimCreateApiConnection("127.0.0.1", 17771, 5000, 10000)

# start simulation
ret = usimpy.UsimStartSim(id, 10000)
print (ret)

## control
control = usimpy.UsimSpeedAdaptMode()
control.expected_speed = 0.8 # m/s
control.steering_angle = 0.0 # angle
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
root_dir = '.2019.11.13/normal_dataset_stpan11.13/'
#root_dir = '.2019.11.13/abnormal_dataset/'
img_dir = root_dir + 'img/'
mkdir(img_dir)
button = False
img_text = np.zeros((400,400,3))
cv2.putText(img_text,"press q to stop,press s to start",(10,200),cv2.FONT_HERSHEY_SIMPLEX,0.7,(254,254,254))


while(1):
    # control vehicle via speed & steer
    ret = usimpy.UsimSetVehicleControlsBySA(id, control)
    # get vehicle post & action
    ret = usimpy.UsimGetVehicleState(id, action)
    # get collision
    ret =usimpy.UsimGetCollisionInformation(id, collision)
    # get RGB image
    ret = usimpy.UsimGetOneCameraResponse(id, 0, image)
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
        img = np.array(image.image[0:480*320*3])
        img = img.reshape((320, 480, 3))
        img_PIL = Image.fromarray(np.uint8(img))
        img_PIL.save(img_dir+str(count)+'.png', 'png')
        # save pose & action
        with open(root_dir + 'action_pose.txt', 'a+') as f:
            f.write(str(count)+',' \
                     +str(action.pose.position.y_val)+','+ str(action.pose.position.z_val)+','+str(action.pose.rotation.x_val)+','+str(action.pose.rotation.y_val)+','\
                     +str(action.pose.rotation.z_val)+','+str(action.steering_angle)+','+ str(action.forward_speed)+'\n')
#           
#            f.write(str(action.steering_angle)+'\n')

#            f.write('time: %d, steer: %.6f, speed: %.6f, time_gap: %d, collision: %d, collision_time: %d' % (
#                action.time_stamp, action.steering_angle, action.forward_speed, (time1-time2), collision.is_collided, collision.time_stamp))
            # auxiliary info
        time1 = action.time_stamp
        print ('time: %d, steer: %.6f, speed: %.6f, time_gap: %d, collision: %d, collision_time: %d' % (
                action.time_stamp, action.steering_angle, action.forward_speed, (time1-time2), collision.is_collided, collision.time_stamp))
        time2 = time1
        count = count + 1
cv2.destroyAllWindows()
# stop simulation
ret = usimpy.UsimStopSim(id)
print (ret)

