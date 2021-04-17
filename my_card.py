# 导入工具包
from imutils import contours
import numpy as np
import cv2
import myutils
from matplotlib import pyplot as plt
# 设置参数
image=cv2.imread('D:/OpenCV/image/card.jpg')
img=cv2.imread('D:/OpenCV/image/pic.png')

# 绘图展示
def cv_show(name,img):
	cv2.imshow(name,img)
	cv2.waitKey(0)
# 灰度图
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv_show('ref_gray',ref)
# 二值图像
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
#cv_show('ref_bi',ref)

# 计算轮廓
#cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图）,cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
#返回的list中每个元素都是图像中的一个轮廓
refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,refCnts,-1,(0,0,255),3) # 轮廓在二值图上得到, 画要画在原图上
#cv_show('img',img)
#print (np.array(refCnts).shape)#输出轮廓数量
refCnts = myutils.sort_contours(refCnts, method="left-to-right")[0] #对轮廓排序，从左到右，从上到下
digits = {}

# 遍历每一个轮廓
for i,c in enumerate(refCnts):#i为索引，c为每个轮廓的终点坐标
	# 计算外接矩形并且resize成合适大小
	(x, y, w, h) = cv2.boundingRect(c)
	roi = ref[y:y + h, x:x + w]#opencv默认先高后长
	roi = cv2.resize(roi, (57, 88))#调大尺寸
	# 每一个数字对应每一个模板
	digits[i] = roi
# 【模板处理流程: 轮廓检测,外接矩形,抠出模板,让模板对应每个数值】

# 【输入图像处理】
# 形态学操作,礼帽+闭操作可以突出明亮区域,但并不是非得礼帽+闭操作
# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

#读取输入图像，预处理
#cv_show('image',image)
image = myutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv_show('image_gray',gray)

#黑帽操作，突出更深色的区域
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel) 
#cv_show('image_blackhat',blackhat) 

#x方向的Sobel算子,实验表明,加y的效果的并不好
gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0,ksize=-1)#ksize=-1相当于用3*3的
gradX = np.absolute(gradX)#计算绝对值
gradY = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=0, dy=1,ksize=-1)
gradY = np.absolute(gradY)
gradXY=cv2.addWeighted(gradX,0.5,gradY,0.5,0)
(minVal, maxVal) = (np.min(gradXY), np.max(gradXY))
gradXY = (255 * ((gradXY - minVal) / (maxVal - minVal)))#归一化
gradXY = gradXY.astype("uint8")
#print (np.array(gradXY).shape)
#cv_show('image_sobel_gradXY',gradXY)

#通过闭操作（先膨胀，再腐蚀）将数字连在一起，将本是4个数字的4个框膨胀成1个框,就腐蚀不掉了
gradXY = cv2.morphologyEx(gradXY, cv2.MORPH_CLOSE, rectKernel) 
#cv_show('image_close',gradXY)
#THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
thresh = cv2.threshold(gradXY, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] 
#cv_show('image_thresh',thresh)

#再来一个闭操作，填补洞点
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
#cv_show('image_close_thresh',thresh)

# 计算轮廓
threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img,cnts,-1,(0,0,255),3) 
cv_show('image_contours',cur_img)
locs = []

# 遍历轮廓
for (i, c) in enumerate(cnts):
	# 计算矩形
	rect=cv2.minAreaRect(c)
	box=np.int0(cv2.boxPoints(rect))
	w,h=int(rect[1][1]+1),int(rect[1][0]+1)
	x,y=box[1][0],box[1][1]
	print((x,y,w,h))
	ar = w / float(h)#利用长宽比进行筛选
	# 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
	if ar > 8.0 and ar < 15.0:
		if (w > 150 and w < 300) and (h > 15 and h < 30):
			#符合的留下来
			locs.append((x, y, w, h))

#cv_show('image_contours',cur_img)
# 将符合的轮廓从左到右排序
locs = sorted(locs, key=lambda x:x[0])
output = []

# 遍历每一个轮廓中的数字
for (i, (gX, gY, gW, gH)) in enumerate(locs):#遍历每一组大轮廓（其中包含4个数字）
	# initialize the list of group digits
	groupOutput = []

	# 根据坐标提取每一个组(4个值)
	group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]#比数字扩大一点点
	#cv_show('group_'+str(i),group)
	# 1.预处理
	group1= cv2.threshold(group, 0, 255,cv2.THRESH_BINARY_INV| cv2.THRESH_OTSU)[1]
	group_name = 'Input_group_' + str(i)
	#cv_show(group_name,group1)
	#2.计算每一组内的轮廓，这样就得到小的数字轮廓了
	digitCnts,hierarchy = cv2.findContours(group1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	#digitCnts= contours.sort_contours(contours,method="left-to-right")[0]
	cv2.drawContours(group,digitCnts,-1,(0,0,225),2)
	cv_show('digitcnts_group',group)

	for (j,c) in enumerate(digitCnts):		# c表示每个小轮廓的终点坐标
		ares=cv2.contourArea(c)
		if ares<30 or ares>100:
			continue
		z=0
		# 找到当前数值的轮廓,resize成合适的的大小
		(x, y, w, h) = cv2.boundingRect(c)	# 外接矩形
		roi = group[y:y + h, x:x + w]		# 在原图中取出小轮廓覆盖区域,即数字
		roi = cv2.resize(roi, (57, 88))
		roi_name = 'roi_'+str(z)
		cv_show(roi_name,roi)
		
		#计算匹配得分,每个数值分别与10个模板数值匹配，计算评分
		scores = []#在单次循环中，scores存的是当前数值与10个模板分别匹配得出的最大值

		# 在模板中计算每一个得分
		# digits的digit正好是数值0,1,...,9;digitROI是每个数值的特征表示
		for (digit, digitROI) in digits.items():
		# 模板匹配，result是结果矩阵
			result = cv2.matchTemplate(roi, digitROI,cv2.TM_CCOEFF)
			(_, score, _, _) = cv2.minMaxLoc(result)#匹配十次，看最高得分是多少
			scores.append(score)

		# 得到最合适的数字
		groupOutput.append(str(np.argmax(scores)))#取十次匹配中的最大值
		z=z+1
	
	cv2.rectangle(image, (gX - 5, gY - 5),
		(gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
	cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
		cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

	# 得到结果
	output.extend(groupOutput)

# 打印结果
print("Credit Card #: {}".format("".join(output)))
cv2.imshow("Image", image)
cv2.waitKey(0)
