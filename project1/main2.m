clear;
clc;

%% 读取图像
x = imread('..\BMP_images\cman256.BMP');
% x = imread('..\BMP_images\lena256.BMP');
figure,subplot(2,2,1), imshow(x), title('Raw figure')

%% 对图像DCT变换  
y=dct2(x);
% figure, imshow(log(abs(dctgrayImage)),[]),title('DCT变换灰度图像'), colormap(gray(4)), colorbar;
subplot(2,2,2), imshow(log(abs(y)),[]),title('DCT Transform');


%% 低通滤波
dctgrayImage=y;
dctgrayImage(abs(dctgrayImage)<0.001)=0; 

%% DCT逆变换  
I=idct2(dctgrayImage)/255;  
subplot(2,2,3),imshow(I), title('Low-pass Filtering and IDCT'); 

%% 高通滤波
dctgrayImage=y;
dctgrayImage(abs(dctgrayImage/255)>5)=0; 

%% DCT逆变换  
I=idct2(dctgrayImage)/255;  
subplot(2,2,4), imshow(I), title('High-pass Filtering and IDCT'); 

