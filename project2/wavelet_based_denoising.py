# @description:
# @author:Jianping Zhou
# @email:jianpingzhou0927@gmail.com
# @Time:2022/11/12 16:09
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
# 小波库
import pywt
import cv2

# 读取图像并转化为灰度图
im = cv2.imread('../BMP_images/lena512.BMP')
im = cv2.resize(im, (512, 512))
img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)

# 对图像进行小波分解
coeffs = pywt.dwt2(img, 'bior1.3')
LL, (LH, HL, HH) = coeffs

# 查看分解结果
plt.subplot(221), plt.imshow(LL, 'gray'), plt.title("LL")
plt.subplot(222), plt.imshow(LH, 'gray'), plt.title("LH")
plt.subplot(223), plt.imshow(HL, 'gray'), plt.title("HL")
plt.subplot(224), plt.imshow(HH, 'gray'), plt.title("HH")
plt.show()
