# @description:
# @author:Jianping Zhou
# @email:jianpingzhou0927@gmail.com
# @Time:2022/11/12 14:41
import cv2
import random
import numpy as np


def pepper_and_salt(img, percentage):
    num = int(percentage * img.shape[0] * img.shape[1])  # 椒盐噪声点数量
    random.randint(0, img.shape[0])
    img2 = img.copy()
    for i in range(num):
        X = random.randint(0, img2.shape[0] - 1)  # 从0到图像长度之间的一个随机整数,因为是闭区间所以-1
        Y = random.randint(0, img2.shape[1] - 1)
        if random.randint(0, 1) == 0:  # 黑白色概率55开
            img2[X, Y] = (255, 255, 255)  # 白色
        else:
            img2[X, Y] = (0, 0, 0)  # 黑色
    return img2


'''
# 自适应中值滤波法
def amp_medianBlur(img, ksize=2):
    # 图像边缘扩展
    # 为保证边缘的像素点可以被采集到，必须对原图进行像素扩展。
    # 一般设置的最大滤波窗口为7，所以只需要向上下左右各扩展3个像素即可采集到边缘像素。
    m, n = img.shape[:2]
    Nmax = 3
    imgn = np.zeros((m + 2 * Nmax, n + 2 * Nmax), dtype=np.uint8)

    imgn[Nmax: (m + Nmax), Nmax: (n + Nmax)] = img[:, :, 0].copy()  # 将原图覆盖在imgn的正中间

    # 下面开始向外扩展，即把边缘的像素向外复制
    imgn[0: Nmax, Nmax: n + Nmax] = img[0: Nmax, 0: n, 0].copy()  # 扩展上边界
    imgn[0: m + Nmax, n + Nmax: n + 2 * Nmax] = imgn[0: m + Nmax, n: n + Nmax].copy()  # 扩展右边界
    imgn[m + Nmax: m + 2 * Nmax, Nmax: n + 2 * Nmax] = imgn[m: m + Nmax, Nmax: n + 2 * Nmax].copy()  # 扩展下边界
    imgn[0: m + 2 * Nmax, 0: Nmax] = imgn[0: m + 2 * Nmax, Nmax: 2 * Nmax].copy()  # 扩展左边界
    re = imgn.copy()  # 扩展之后的图像

    # 得到不是噪声点的中值
    for i in range(Nmax, m + Nmax + 1):
        for j in range(Nmax, n + Nmax + 1):
            r = 1  # 初始向外扩张1像素，即滤波窗口大小为3
            while r != Nmax + 1:  # 当滤波窗口小于等于7时（向外扩张元素小于4像素）

                W = imgn[i - r - 1:i + r, j - r - 1: j + r].copy()
                Imin, Imax = np.min(W), np.max(W)  # 最小灰度值 # 最大灰度值

                # 取中间值
                window = np.sort(W, axis=None)

                if len(window) % 2 == 1:
                    Imed = window[len(window) // 2]

                else:
                    Imed = int((window[len(window) // 2] + window[len(window) // 2 + 1]) / 2)
                if Imin < Imed and Imed < Imax:  # 如果当前窗口中值不是噪声点，那么就用此次的中值为替换值
                    break
                else:
                    r = r + 1  # 否则扩大窗口，继续判断，寻找不是噪声点的中值
                # 判断当前窗口内的中心像素是否为噪声，是就用前面得到的中值替换，否则不替换
            if Imin < imgn[i, j] and imgn[i, j] < Imax:  # 如果当前这个像素不是噪声，原值输出
                re[i, j] = imgn[i, j].copy()
            else:  # 否则输出邻域中值
                re[i, j] = Imed
    return re


img_amp_median = amp_medianBlur(img2)
cv2.imshow('lena_ampmedian', img_amp_median)
'''


def highPassFiltering(img, size):  # 传递参数为傅里叶变换后的频谱图和滤波尺寸
    h, w = img.shape[0:2]  # 获取图像属性
    h1, w1 = int(h / 2), int(w / 2)  # 找到傅里叶频谱图的中心点
    img[h1 - int(size / 2):h1 + int(size / 2),
    w1 - int(size / 2):w1 + int(size / 2)] = 0  # 中心点加减滤波尺寸的一半，刚好形成一个定义尺寸的滤波大小，然后设置为0
    return img


def lowPassFiltering(img, size):  # 传递参数为傅里叶变换后的频谱图和滤波尺寸
    h, w = img.shape[0:2]  # 获取图像属性
    h1, w1 = int(h / 2), int(w / 2)  # 找到傅里叶频谱图的中心点
    img2 = np.zeros((h, w), np.uint8)  # 定义空白黑色图像，和傅里叶变换传递的图尺寸一致
    img2[h1 - int(size / 2):h1 + int(size / 2),
    w1 - int(size / 2):w1 + int(size / 2)] = 1  # 中心点加减滤波尺寸的一半，刚好形成一个定义尺寸的滤波大小，然后设置为1，保留低频部分
    img3 = img2 * img  # 将定义的低通滤波与传入的傅里叶频谱图一一对应相乘，得到低通滤波
    return img3


if __name__ == "__main__":
    img = cv2.imread("../BMP_images/lena512.BMP")
    img2 = pepper_and_salt(img, 0.04)  # 百分之4的椒盐噪音
    cv2.imshow("lena", img)
    cv2.imwrite('./results/lena.png', img)
    cv2.imshow("lena_pepper_and_salt", img2)
    cv2.imwrite('./results/lena_salt_pepper.png', img2)
    cv2.waitKey(0)

    # 均值滤波和中值滤波
    img_mean = cv2.blur(img2, (3, 3))  # 均值滤波
    img_median = cv2.medianBlur(img2, 3)  # 中值滤波

    cv2.imshow("lena_mean", img_mean)
    cv2.imwrite('./results/lena_mean_filter.png', img_mean)
    cv2.imshow("lena_median", img_median)
    cv2.imwrite('./results/lena_median_filter.png', img_median)
    cv2.waitKey(0)
