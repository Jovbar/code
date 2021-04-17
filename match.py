import cv2 
import numpy as np

img1 = cv2.imread('D:/OpenCV/image/box.png', 0)
img2 = cv2.imread('D:/OpenCV/image/box_in_scene.png', 0)
def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
cv_show('img1',img1)
cv_show('img2',img2)


# 第一步：构造sift，求解出特征点和sift特征向量（两个返回值）
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 第二步：构造BFMatcher()蛮力匹配，匹配sift特征向量距离最近对应组分
# crossCheck 表示两个特征点要互相匹，
# 例如A中的第i个特征点与B中的第j个特征点最近的,并且B中的第j个特征点到A中的第i个特征点也是 
# NORM_L2: 归一化数组的(欧几里德距离)，如果其他特征计算方法需要考虑不同的匹配计算方式
bf = cv2.BFMatcher(crossCheck=True)

# 获得匹配的结果【1 对 1 匹配】
matches = bf.match(des1, des2)

#第三步：对匹配的结果按照距离进行排序操作
matches = sorted(matches, key=lambda x: x.distance)

# 第四步：使用cv2.drawMacthes进行画图操作
# 只画前10个点
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None,flags=2)
cv_show('img3',img3)

# 【k对最佳匹配】
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)#可以进行一对多操作
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

# good所有点都显示,就会有很多连线了
img4 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
cv_show('img4',img4)