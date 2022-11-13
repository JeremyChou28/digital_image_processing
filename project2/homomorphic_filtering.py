# @description:
# @author:Jianping Zhou
# @email:jianpingzhou0927@gmail.com
# @Time:2022/11/12 15:59
import os
import cv2
from PIL import Image
import numpy as np
from salt_pepper_noise import pepper_and_salt


def homomorphic_filter(src, d0=10, r1=0.5, rh=2, c=4, h=2.0, l=0.5):
    gray = src.copy()
    if len(src.shape) > 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)
    rows, cols = gray.shape
    gray_fft = np.fft.fft2(gray)
    gray_fftshift = np.fft.fftshift(gray_fft)
    dst_fftshift = np.zeros_like(gray_fftshift)
    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2))
    D = np.sqrt(M ** 2 + N ** 2)
    Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + r1
    dst_fftshift = Z * gray_fftshift
    dst_fftshift = (h - l) * dst_fftshift + l
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)
    dst = np.real(dst_ifft)
    dst = np.uint8(np.clip(dst, 0, 255))
    return dst


if __name__ == "__main__":
    img = cv2.imread("../BMP_images/lena512.BMP")
    # img = Image.open(path)
    # Img = img.convert('L')
    # img = np.array(img)
    # print(img, img.shape)
    img2 = pepper_and_salt(img, 0.04)  # 百分之4的椒盐噪音
    img_new = homomorphic_filter(img2)
    print("new img shape is {}".format(img_new.shape))
    cv2.imshow("lena_homomorphic", img_new)
    cv2.imwrite("./results/lena_homomorphic.png", img_new)
    cv2.waitKey(0)
    # cv2.imwrite("2.png", img_new)
