#%%
import os
import sys
import shutil
import numpy as np
import cv2 as cv
#%%
src_dir = "D:/dataset/Chunk_1/Chunk_1/"
dst_dir = "D:/dataset/comma_c12_uisee_104_105/"
if not os.path.isdir(dst_dir):
    os.mkdir(dst_dir)
if not os.path.isdir(dst_dir+"img"):
    os.mkdir(dst_dir+"img")
#%%
total_dirs = []
first_sub_dir = os.listdir(src_dir)[1:]
for i in first_sub_dir:
    second_sub_dir = os.listdir(src_dir+i)
    second_sub_dir.sort(key = lambda x: int(x))
    for j in second_sub_dir:
        total_dirs.append(src_dir+i+"/"+j)
#%%
f = open(dst_dir+"comma.txt",'w')
count = 0
for i in total_dirs[104:106]:
    speed = np.load(i +"/"+ 'processed_log/CAN/speed/value')
    angle = np.load(i +"/"+ 'processed_log/CAN/steering_angle/value')
    if len(speed) < len(angle):
        loop_len = speed.shape[0]
    else:
        loop_len = angle.shape[0]
    for j in range(loop_len):
        f.write(str(count)+","+str(angle[j])\
            +","+str(speed[j,0])+"\n")
        count += 1
f.close()
#%%
count = 0
for i in total_dirs[104:106]:
    cap = cv.VideoCapture(i+"/"+"video.hevc")
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        frame = cv.resize(frame, (480,320))
        # Display the resulting frame
        cv.imwrite(dst_dir+"img/"+str(count)+".jpg",frame)
        count+=1
# When everything done, release the capture
cap.release()
#%% 临时

cap = cv.VideoCapture(total_dirs[115]+"/"+"video.hevc")
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.namedWindow("frame",cv.WINDOW_NORMAL)
    frame = cv.resize(frame, (480,320))
    frame = frame[80:220,:,:]
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

# %%
