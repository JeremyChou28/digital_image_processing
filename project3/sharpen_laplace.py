# @description:
# @author:Jianping Zhou
# @email:jianpingzhou0927@gmail.com
# @Time:2022/11/29 9:30
import cv2
import numpy as np
from skimage import io

# 导入图片
raw_img = cv2.imread("../BMP_images/lena512.BMP")
cv2.imshow('raw_image', raw_img)

# 转换灰度
# gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gimg = raw_img

# 高斯模糊化
# dst = cv2.GaussianBlur(img,ksize=(5,5),sigmaX=0,sigmaY=0)
# 创建毛玻璃特效
# 参数2：高斯核的宽和高（建议是奇数）
# 参数3：x和y轴的标准差
img = cv2.GaussianBlur(gimg, (11, 11), 0)
cv2.imshow('GaussianBlur_image', img)
io.imsave('./results/GaussianBlur_image.png', img)

# 拉普拉斯算子锐化
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 定义拉普拉斯算子
dst = cv2.filter2D(img, -1, kernel=kernel)  # 调用opencv图像锐化函数

# sobel算子锐化
# 对x方向梯度进行sobel边缘提取
x = cv2.Sobel(gimg, cv2.CV_64F, 1, 0)
# 对y方向梯度进行sobel边缘提取
y = cv2.Sobel(gimg, cv2.CV_64F, 0, 1)
# 对x方向转回uint8
absX = cv2.convertScaleAbs(x)
# 对y方向转会uint8
absY = cv2.convertScaleAbs(y)
# x，y方向合成边缘检测结果
dst1 = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
# 与原图像堆叠
res = dst1 + gimg

# 测试
# print("dstshape:",dst1)
# print("resshape:",res)

# 按要求左右显示原图与拉普拉斯处理结果
# result1 = np.hstack([raw_img, img, dst])
result1 = dst
cv2.imshow('lapres', result1)
io.imsave('./results/lapres.png', result1)

# 按要求左右显示原图与sobel处理结果
# result2 = np.hstack([raw_img, img, res])
result2 = res
cv2.imshow('sobelres', result2)
io.imsave('./results/sobelres.png', result2)

# 去缓存
cv2.waitKey(0)
cv2.destroyAllWindows()
