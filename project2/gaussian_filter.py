# @description:
# @author:Jianping Zhou
# @email:jianpingzhou0927@gmail.com
# @Time:2022/11/12 16:12
import cv2
import numpy as np
from salt_pepper_noise import pepper_and_salt


def gaussian_filter(img, K_size=3, sigma=1.3):
    if len(img.shape) == 3:

        H, W, C = img.shape

    else:

        img = np.expand_dims(img, axis=-1)

        H, W, C = img.shape

    ## Zero padding

    pad = K_size // 2

    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=float)

    out[pad: pad + H, pad: pad + W] = img.copy().astype(float)

    ## prepare Kernel

    K = np.zeros((K_size, K_size), dtype=float)

    for x in range(-pad, -pad + K_size):

        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))

    K /= (2 * np.pi * sigma * sigma)

    K /= K.sum()

    tmp = out.copy()

    # filtering

    for y in range(H):

        for x in range(W):

            for c in range(C):
                out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])

    out = np.clip(out, 0, 255)

    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

    return out


if __name__ == "__main__":
    img = cv2.imread("../BMP_images/lena512.BMP")
    # img = Image.open(path)
    # Img = img.convert('L')
    # img = np.array(img)
    # print(img, img.shape)
    img2 = pepper_and_salt(img, 0.04)  # 百分之4的椒盐噪音
    img_new = gaussian_filter(img2)
    print("new img shape is {}".format(img_new.shape))
    cv2.imshow("lena_gaussian", img_new)
    cv2.imwrite("./results/lena_gaussian.png", img_new)
    cv2.waitKey(0)
