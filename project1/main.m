clear;
clc;

%% 读取图像
x = imread('.\BMP_images\cman256.BMP');
% x = imread('.\BMP_images\lena256.BMP');
figure(1),imshow(x), title('Raw figure'),saveas(gcf,'Raw_figure1','jpg')
figure(2),subplot(2,2,1), imshow(x), title('Raw figure')

%% 傅里叶变换
y = fft2(x);       % 二维傅里叶变换
y = fftshift(y);   % 将零频分量移到频谱中心
y_abs = abs(y);    % 对复数进行模运算 
z = log(y_abs+1);  % 映射到小范围区间
% 根据z中的像素值范围对显示进行转换，将z中的最小值显示为黑色，最大值显示为白色。
subplot(2,2,2), imshow(z, []), title('DFT Transform');

% %% 低通滤波器
% LPF = ones(size(y));        % 初始化低通滤波器
% threshold_low = 50;         % 设置截至频率
% [row, col] = size(y);       % 得到图像y的高度和宽度
% for i = 1: row
%     for j = 1: col
%         % 计算频率
%         if sqrt((i-row/2)^2+(j-col/2)^2) > threshold_low
%             % 将低通滤波器中高于截止频率的频率变为0
%             LPF(i,j) = 0;
%         end
%     end
% end
% y_LPF = y.*LPF;             % 低通滤波
% y_LPF = ifftshift(y_LPF);   % 逆零频平移
% x_LPF = ifft2(y_LPF);       % 傅里叶逆变换
% x_LPF = uint8(real(x_LPF)); % 转换成实数，并将double转成uint8
% figure, imshow(x_LPF), title('低通滤波')

%% 计算频率，频率 = 当前点(i,j) 到 中心点(row/2, col/2)的距离
[row, col] = size(y);
distance = zeros(row, col);
for i = 1: row
    for j = 1: col
        % 计算向量的模长，即两点间距离
        distance(i,j) = norm([i-row/2 j-col/2]);
    end
end

%% 低通滤波 (Low Pass Filtering, LPF)
y_LPF = y;                  % 复制一个正常的y
threshold_low = 50;         % 设置阈值
% 过滤频率高于阈值的点
y_LPF(distance > threshold_low) = 0;
y_LPF = ifftshift(y_LPF);   % 逆零频平移
x_LPF = ifft2(y_LPF);       % 傅里叶逆变换
x_LPF = uint8(real(x_LPF)); % 转换成实数，并将double转成uint8
subplot(2,2,3),imshow(x_LPF), title('Low-pass Filtering and IDFT')

%% 高通滤波 (HPF, High Pass Filtering)
y_HPF = y;
threshold_high = 30;
y_HPF(distance < threshold_high) = 0;
y_HPF = ifftshift(y_HPF);
x_HPF = ifft2(y_HPF);
x_HPF = uint8(real(x_HPF));
subplot(2,2,4), imshow(x_HPF), title('High-pass Filtering and IDFT')