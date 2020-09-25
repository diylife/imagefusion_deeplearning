% �ֵ�ѧϰ��K-SVD
function train_dictionary
for i=1:21
disp(num2str(i));
index = i;
path1 = ['./IV_images/IR',num2str(index),'.png'];
path2 = ['./IV_images/VIS',num2str(index),'.png'];

source_image1 = imread(path1);
source_image2 = imread(path2);

I1 = im2double(source_image1);
I2 = im2double(source_image2);

[m,n] = size(I1);

unit = 7;
step = 3;
step2 = (unit-1)/2;

row_unit = unit*unit;

disp(strcat('��ʼ����ֿ�����'));
count = 0;
for i=(1+step2):step:(m-step2)
    for j=(1+step2):step:(n-step2)
        count = count+1;
        patch1 = I1((i-3):(i+3),(j-3):(j+3));
        patch2 = I2((i-3):(i+3),(j-3):(j+3));
        
        Vi1(:, count) = patch1(:);
        Vi2(:, count) = patch2(:);
    end
end
disp(strcat('��������ֿ�����'));

patch_num = count;
Vc = zeros(row_unit, patch_num*2);%��������

Vc(:,1:patch_num) = Vi1;
Vc(:,(patch_num+1):2*patch_num) = Vi2;

% KSVDѵ�����ֵ�
dic_size = 256;
k = 16;

disp('KSVD-�����ֵ� D');
tic
params.data = Vc;
params.Tdata = k;
params.dictsize = dic_size;
params.iternum = 50;
params.memusage = 'high';
[D,X,err] = ksvd(params,'');
toc
disp('KSVD-�����ֵ� ����');

%�����ֵ�
D_path = './dictionary/';
D_name = [D_path, 'D_unit7_im',num2str(index),'.dat'];
save(D_name,'D');

clear Vi1;
clear Vi2;
end
end








