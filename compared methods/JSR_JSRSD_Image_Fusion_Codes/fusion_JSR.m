% C.H. Liu, Y. Qi, W.R. Ding.
% Infrared and visible image fusion method based on saliency detection in sparse domain[J]. Infrared Physics & Technology.
% joint sparse representation, l1+weight

addpath(genpath('./dictionary'));
addpath(genpath('./ompbox10'));
addpath(genpath('./ksvdbox13'));
h=waitbar(0,'����ʼ��ȴ���');
for inde=1:21
    
    index = inde;
    disp(num2str(index));
    
    matName = strcat('D_unit7_im',num2str(index));
    matName = strcat(matName, '.dat');
    load(matName, '-mat');
    
    path1 = ['./IV_images/IR',num2str(index),'.png'];
    path2 = ['./IV_images/VIS',num2str(index),'.png'];
    fused_path = ['./fused/fused',num2str(index),'_jsr.png'];
    
    source_image1 = imread(path1);
    source_image2 = imread(path2);
    
    I1 = im2double(source_image1);
    I2 = im2double(source_image2);
    
    [m,n] = size(I1);
    
    unit = 7;
    step = 1;
    
    row_unit = unit*unit;
    
    disp(strcat('��ʼ����ֿ�����'));
    count = 0;
    for i=4:step:(m-3)
        for j=4:step:(n-3)
            count = count+1;
            patch1 = I1((i-3):(i+3),(j-3):(j+3));
            patch2 = I2((i-3):(i+3),(j-3):(j+3));
            
            Vi1(:, count) = patch1(:);
            Vi2(:, count) = patch2(:);
        end
    end
    disp(strcat('��������ֿ�����'));
    
    patch_num = count;
    
    % KSVDѵ�����ֵ�
    dic_size = 256;
    k = 16;
    
    V_Joint = zeros(2*row_unit, patch_num);
    V_Joint(1:(row_unit),:) = Vi1;
    V_Joint((row_unit+1):(2*row_unit),:) = Vi2;
    
    D_Joint = zeros(2*row_unit, dic_size*3);
    D_Joint(1:row_unit,1:dic_size) = D/sqrt(2);
    D_Joint(1:row_unit,(dic_size+1):2*dic_size) = D;
    D_Joint((row_unit+1):2*row_unit,1:dic_size) = D/sqrt(2);
    D_Joint((row_unit+1):2*row_unit,(dic_size*2+1):3*dic_size) = D;
    
    disp('OMP-���ϵ��');
    tic
    C = zeros(dic_size*3, count);
    for i=1:count
        c = omp(D_Joint, V_Joint(:,i),[], k);
        C(:,i) = c;
    end
    toc
    disp('OMP-���ϵ�� ����');
    
    coe_c = (C(1:dic_size,:));
    coe_u1 = C((dic_size+1):2*dic_size,:);
    coe_u2= C((2*dic_size+1):3*dic_size,:);
    
    % V_com = D*coe_c;
    % V_u1  = D*coe_u1;
    % V_u2  = D*coe_u2;
    
    %��ϵ�������ں�
    disp('�ںϿ�ʼ');
    tic
    % m_unit = floor(m/unit);
    % n_unit = floor(n/unit);
    coe_fusion = zeros(dic_size, count);
    for i=1:count
        coe_index = i;
        n1 = norm(coe_u1(:,coe_index),1);
        n2 = norm(coe_u2(:,coe_index),1);
        w1 = n1/max(n1,n2);
        w2 = n2/max(n1,n2);
        coe_fusion(:,coe_index) = coe_c(:,coe_index)+w1*coe_u1(:,coe_index)+w2*coe_u2(:,coe_index);
        
    end
    toc
    disp('�ںϽ���');
    V_fusion = D*coe_fusion;
    
    fusion = zeros(m,n);
    countt = 0;
    for i=4:step:(m-3)
        for j=4:step:(n-3)
            countt = countt+1;
            patch1 = V_fusion(:, countt);
            pr = reshape(patch1,[unit,unit]);
            temp = fusion((i-3):(i+3),(j-3):(j+3));
            fusion((i-3):(i+3),(j-3):(j+3))=(pr+temp)/2;
        end
    end
    % figure;imshow(fusion);
    
    imwrite(fusion,fused_path,'png');
    
    clear Vi1;
    clear Vi2;
    waitbar(index/21);
end
close(h);



