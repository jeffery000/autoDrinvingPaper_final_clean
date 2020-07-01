from keras.models import load_model    
import numpy as np
import cv2
import os
import math
model_name = "log/5_19ep_calar_nvNet_2pred_test_half/KivlNet_part2_30.h5"
original_dir = "D:/dataset/carla527_test_train_1_3_m/img/"
dst_dir = "D:/dataset/carla527_test_train_1_3_m/img_plot/"
model = load_model(model_name)
file_names = os.listdir(original_dir)
file_names.sort(key = lambda x: int(x[:-4]))
file_names = file_names[:3169]
for i in file_names:
    img = cv2.imread(original_dir+i)
    img_batch = cv2.resize(img,(160,80))/255
    img_batch = np.expand_dims(img_batch, axis=0)
    pred = model.predict(img_batch)
    print("***",pred)

    alphaReserve = 0.8
    BChannel = 255
    GChannel = 0
    RChannel = 0
    yMin = 227
    yMax = 307
    xMin = 10
    xMax = 220

    img[yMin:yMax, xMin:xMax, 0] = img[yMin:yMax, xMin:xMax, 0] * alphaReserve + BChannel * (1 - alphaReserve)
    img[yMin:yMax, xMin:xMax, 1] = img[yMin:yMax, xMin:xMax, 1] * alphaReserve + GChannel * (1 - alphaReserve)
    img[yMin:yMax, xMin:xMax, 2] = img[yMin:yMax, xMin:xMax, 2] * alphaReserve + RChannel * (1 - alphaReserve)
    
    yMin1 = 240
    yMax1 = 260
    xMin1 = 230
    xMax1 = 600

    img[yMin1:yMax1, xMin1:xMax1, 0] = img[yMin1:yMax1, xMin1:xMax1, 0] * alphaReserve + BChannel * (1 - alphaReserve)
    img[yMin1:yMax1, xMin1:xMax1, 1] = img[yMin1:yMax1, xMin1:xMax1, 1] * alphaReserve + GChannel * (1 - alphaReserve)
    img[yMin1:yMax1, xMin1:xMax1, 2] = img[yMin1:yMax1, xMin1:xMax1, 2] * alphaReserve + RChannel * (1 - alphaReserve)

    yMin2 = 270
    yMax2 = 290
    xMin2 = 230
    xMax2 = 600

    img[yMin2:yMax2, xMin2:xMax2, 0] = img[yMin2:yMax2, xMin2:xMax2, 0] * alphaReserve + BChannel * (1 - alphaReserve)
    img[yMin2:yMax2, xMin2:xMax2, 1] = img[yMin2:yMax2, xMin2:xMax2, 1] * alphaReserve + GChannel * (1 - alphaReserve)
    img[yMin2:yMax2, xMin2:xMax2, 2] = img[yMin2:yMax2, xMin2:xMax2, 2] * alphaReserve + RChannel * (1 - alphaReserve)

    steer = pred[0][0]
    speed = pred[0][1]
    
    img[yMin1:yMax1, int(415+200*steer-10):int(415+200*steer+10)] = [160,160,160]
    img[yMin1:yMax1, int(413):int(417)] = [197, 205,122]
    img[yMin2:yMax2, int(415+10*(speed-25)-10):int(415+10*(speed-25)+10)] = [160,160,160]
    img[yMin2:yMax2, int(413):int(417)] = [197, 205,122]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Steer:%.2f"%steer, (25, 255), font, 1, (255, 255, 255), 2)
    cv2.putText(img, "Speed:%.2f"%speed,(25, 290), font, 1, (255, 255, 255), 2)
    cv2.putText(img, "0", (405, 230), font, 1, (255, 255, 255), 2)
    cv2.putText(img, "25km/s",(360, 315), font, 1, (255, 255, 255), 2)
    # k = 100
    # if steer>0:
    #     steer_sqrt = math.sqrt(abs(steer))
    # else:
    #     steer_sqrt = -math.sqrt(abs(steer))
    # x = np.array([320,320+0.3*steer_sqrt*k,320+steer_sqrt*k])
    # y = np.array([200,150,100])
    # poly = np.poly1d(np.polyfit(x, y, 2))


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
    # cv2.imwrite(dst_dir+i,img)
    cv2.imshow("img",img)
    if cv2.waitKey(30)==ord('q'):
        break
cv2.destroyAllWindows()
        