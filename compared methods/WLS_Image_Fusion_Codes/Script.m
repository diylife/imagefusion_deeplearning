% This code is in association with the following paper
% "Ma J, Zhou Z, Wang B, et al. Infrared and visible image fusion based on visual saliency map and weighted least square optimization[J].
% Infrared Physics & Technology, 2017, 82:8-17."
% Authors: Jinlei Ma, Zhiqiang Zhou, Bo Wang, Hua Zong
% Code edited by Jinlei Ma, email: majinlei121@163.com

clear all;
close all;
h=waitbar(0,'程序开始请耐心等待…');
for i=1:21
    index = i;
    
    path1 = ['./IV_images/VIS',num2str(index),'.png'];
    path2 = ['./IV_images/IR',num2str(index),'.png'];
    fused_path = ['./fused/fused',num2str(index),'_wls.png'];
    
    % I1 is a visible image, and I2 is an infrared image.
    I1 = imread(path1);
    I2 = imread(path2);
    
    I1 = im2double(I1);
    I2 = im2double(I2);
    
    % figure;imshow(I1);
    % figure;imshow(I2);
    tic
    fused = WLS_Fusion(I1,I2);
    toc
    
    % figure;imshow(fused);
    imwrite(fused,fused_path,'png');
    waitbar(i/21);
end
close(h);
