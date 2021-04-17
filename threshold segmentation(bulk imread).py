# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:29:26 2020

@author: Administrator
"""
import os # 处理目录(文件夹的)一个python 库
import cv2

hsil_path = 'D:/practice/HSIL'
hsil_names = os.listdir(hsil_path) # 得到HSIL目录下的文件名， 列表的形式

for j,name in enumerate(hsil_names):
    img = os.path.join(hsil_path, name) # 得到图片对应的完整路径
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      # 转为灰度图
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)     #图像去噪，高斯滤波
    ret,thresh = cv2.threshold(blurred,170,255,cv2.THRESH_BINARY_INV)#通过阈值确定轮廓
    #plt.imshow(thresh,'gray')
    cnts,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours=cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
    count = 0
    for i, contour in enumerate(cnts):
        ares = cv2.contourArea(contour)  # 计算包围形状的面积
        if ares < 300:  # 过滤面积小于300的形状
            continue
        count += 1
        x,y,w,h = cv2.boundingRect(contour)
        if w<h:#得到截取后的图像
            w=h #正方形框架
            rotated_canvas = img[y:y+w+1,x:x+h+1]
        else:
            h=w
            rotated_canvas = img[y:y+h+1,x:x+w+1]
        x1,y1=rotated_canvas.shape[:2]
        #if x1!=y1:
            #continue
        image= cv2.rectangle(contours,(x,y), (x+w,y+h), (255, 0, 0), 1) 
        #print("cell #{}".format(count))
        #cv2.imwrite("E:/imgs/1/%s_{}.jpg".format(count)%name, rotated_canvas)

#image, contours, hierarchy = cv.findContours( image, mode, method[, contours[, hierarchy[, offset]]] )
#摘自OpenCV 3.4.11 官方文档、

#contours, hierarchy = cv.findContours( image, mode, method[, contours[, hierarchy[, offset]]] )
#摘自OpenCV 4.0.1 官方文档
    
