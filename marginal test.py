# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:30:47 2020

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
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)     #图像去噪，高斯滤波
    edged=cv2.Canny(blurred,20,130)
    cnts,hierarchy = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contours=cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
    count = 0
    for i, contour in enumerate(cnts):
        ares = cv2.contourArea(contour)  # 计算包围形状的面积
        if ares < 15:  # 过滤面积小于300的形状
            continue
        count += 1
        x,y,w,h = cv2.boundingRect(contour)
        if w<h:#得到截取后的图像
            w=h #正方形框架
            rotated_canvas = img[y:y+w+1,x:x+h+1]
        else:
            h=w #正方形框架
            rotated_canvas = img[y:y+h+1,x:x+w+1]
        #x1,y1=rotated_canvas.shape[:2]
        #if x1!=y1:
            #continue
        image= cv2.rectangle(contours,(x,y), (x+w,y+h), (255, 0, 0), 1) 
        #print("cell #{}".format(count))
        #cv2.imwrite("E:/imgs/1/%s_{}.jpg".format(count)%name, rotated_canvas)
        cv2.imwrite('D:/practice/pics(1)/{}.jpg'.format(j),image)
    
