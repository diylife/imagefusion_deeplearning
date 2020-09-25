
% Load dictionary
load('dict.mat');  


for i=1:21
index = i;

path1 = ['./CSR_Fusion_Author/IV_images/IR',num2str(index),'.png'];
path2 = ['./CSR_Fusion_Author/IV_images/VIS',num2str(index),'.png'];
fused_path = ['./CSR_Fusion_Author/fused/fused',num2str(index),'_ConvSR.png'];

% Load images
% A=imread('sourceimages/s01_1.tif');
% B=imread('sourceimages/s01_2.tif');
A=imread(path1);
B=imread(path2);

% figure,imshow(A)
% figure,imshow(B)

%key parameters
lambda=0.01; 
flag=1; % 1 for multi-focus image fusion and otherwise for multi-modal image fusion

%CSR-based fusion
tic
F=CSR_Fusion(A,B,D,lambda,flag);
toc

% figure,imshow(F);
imwrite(F,fused_path);

end
