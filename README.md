<!--
 * @Description:
 * @Author: Jianping Zhou
 * @Email: jianpingzhou0927@gmail.com
 * @Date: 2022-12-02 18:46:15
-->

# Digital Image Processing

This is the digital image processing course projects of SJTU.

## Data

`BMP_images` : Some classic grayscale images in digital image processing. The image files' format is bmp.

## Project1

![image1](https://github.com/JeremyChou28/digital_image_processing/blob/main/imgs/image1.png)

`project1/main.m`: use DFT/IDFT, low-pass filtering and high-pass filtering

`project1/main2.m`: use DCT/IDCT, low-pass filtering and high-pass filtering

## Project2

![image2](https://github.com/JeremyChou28/digital_image_processing/blob/main/imgs/image2.png)

`project2/results`：

- `lena.png`: raw figure
- `lena.gaussian.png`: the image processed by gaussian filtering
- `lena_high_pass_filtering.png`: the image processed by high-pasa filtering
- `lena_homomorphic.png`: the image processed by homomorphic filtering
- `lena_low_pass_filtering.png`: the image processed by low-pass filtering
- `lena_mean_filter.png`: the image processed by mean filtering
- `lena_median_filter.png`: the image processed by median filtering
- `lena_salt_pepper.png`: the image after adding salt-and-pepper noise

`project2/gaussian_filter.py`: use guassian filtering

`project2/homomorphic_filtering.py`: use homomorphic filtering

`project2/low_high_pass_filter.py`: use low-pass and high-pass filtering

`project2/salt_pepper_noise.py`: add salt-and-pepper noise

`project2/wavelet_based_denoising.py`: use wavelet-based denoising algorithm

## Project3

![image3](https://github.com/JeremyChou28/digital_image_processing/blob/main/imgs/image3.png)

`project3/results`：

- `lena.png`: raw figure
- `GaussianBlur_image.png`: the image processed by gaussian blurring
- `lapres.png`: the blurred image sharpened by the laplacian operator
- `sobelres.png`: the blurred image sharpened by the Sobel operator

`project3/sharpen_gradient.py`: use gradient method to sharpen image

`project3/sharpen_laplace.py`: use laplacian operator and sobel operator to sharpen image

## Project4

![image4](https://github.com/JeremyChou28/digital_image_processing/blob/main/imgs/image4.png)

`project4/images`：

- `book.png`: a book image
- `person.png`: an image of a person stand nearby window

`project4/results`：

- `book/gray_book.png`: gray image of book
- `book/segmented_region_book.png`: use region-growth segmentation method for book image
- `book/segmented_threshold_book.png`: use threshold-based segmentation method for book image
- `person/gray_person.png`: gray image of person
- `person/segmented_region_person.png`: use region-growth segmentation method for person image
- `person/segmented_threshold_person.png`: use threshold-based segmentation method for person image

`project4/segmentation_book.ipynb`: use threshold-based and region-growth segmentation method to segment book image

`project4/segmentation_person.ipynb`: use threshold-based and region-growth segmentation method to segment person image
