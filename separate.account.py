# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:16:13 2020

@author: WJR
""" 

import numpy as np
import cv2

img=cv2.imread('E:/images/03.jpg')
#BGR转化至HSV
img_1=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#设定阈值
lower_green=np.array([26,43,46])
upper_green=np.array([99,255,255])
lower_red=np.array([160,43,46])
upper_red=np.array([180,255,255])
#根据阈值建立掩膜
mask1=cv2.inRange(img_1,lower_green,upper_green)
mask2=cv2.inRange(img_1,lower_red,upper_red)
#对原图像和掩膜进行位运算
res1=cv2.bitwise_and(img,img,mask=mask1)
res2=cv2.bitwise_and(img,img,mask=mask2)
masks=[mask1,mask2]
images=[res1,res2]
for j in range(2):
    (_,cnts,_) = cv2.findContours(masks[j],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    count = 0 
    for i,contour in enumerate(cnts):
        ares=cv2.contourArea(contour)
        count+=1
        x,y,w,h=cv2.boundingRect(contour)
        image=cv2.putText(images[j],str(count),(x+w,y),cv2.FONT_HERSHEY_COMPLEX,0.4,(255,255,255),1)
        cv2.imwrite('F:/@Deeplearning/{}.jpg'.format(j),image)
