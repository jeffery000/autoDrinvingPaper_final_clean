#%%
import numpy as np 
import cv2

# %%
img = cv2.imread(r"D:\dataset\yushikeji\train_seg\0.tiff")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,img = cv2.threshold(img,10,255,cv2.THRESH_BINARY)
cv2.imshow("img",img)
if cv2.waitKey()==ord('q'):
    cv2.destroyAllWindows()

# %%
