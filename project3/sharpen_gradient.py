# @description:
# @author:Jianping Zhou
# @email:jianpingzhou0927@gmail.com
# @Time:2022/11/28 23:13
from PIL import Image
import numpy as np


def SobelFilter(src, dst, filter_kind=1, padding="replicate"):
    imarray = np.array(Image.open(src), dtype='double')
    # print(imarray.dtype)
    height, width = imarray.shape
    new_arr = np.zeros((height, width), dtype="uint16")

    filter1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    if filter_kind == 1:
        filter = filter1
    elif filter_kind == 2:
        filter = filter2

    k = filter.shape[0]

    for i in range(height):
        for j in range(width):
            total = 0
            for n in range(pow(k, 2)):
                '''
                k = 3, n = 0, 1, 2 ..., 8, a = -1, 0, 1, b = -1, 0, 1
                k = 5, n = 0, 1, 2, 3 ..., 24, a = -2, -1, 0, 1, 2
                '''
                a, b = int(n // k - (k - 1) / 2), int(n % k - 1)
                # filter_value
                aa, bb = int(n // k), int(n % k)
                f_value = filter[aa, bb]
                if i + a <= 0:
                    if j + b <= 0:
                        total += imarray[0, 0] * f_value
                    elif j + b >= width - 1:
                        total += imarray[0, -1] * f_value
                    else:
                        total += imarray[0, j + b] * f_value
                elif i + a >= height - 1:
                    if j + b <= 0:
                        total += imarray[-1, 0] * f_value
                    elif j + b >= width - 1:
                        total += imarray[-1, -1] * f_value
                    else:
                        total += imarray[-1, j + b] * f_value
                else:
                    if j + b <= 0:
                        total += imarray[i + a, 0] * f_value
                    elif j + b >= width - 1:
                        total += imarray[i + a, -1] * f_value
                    else:
                        total += imarray[i + a, j + b] * f_value
            new_arr[i, j] = abs(total)

    max = np.max(new_arr)
    min = np.min(new_arr)
    final_arr = np.zeros((height, width), dtype="uint8")
    for i in range(height):
        for j in range(width):
            final_arr[i, j] = 255 * (new_arr[i, j] - min) / (max - min)

    final_im = Image.fromarray(final_arr)
    final_im.save(dst)


# print("Suceess.")

src = "../BMP_images/lena512.BMP"
dst1 = "results/lena_sharpen_gradient.jpg"

SobelFilter(src, dst1, 1)
# SobelFilter(src, dst2, 2)
