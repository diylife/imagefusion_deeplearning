net = load('G:\BaiduNetdiskDownload\imagenet-resnet-152-dag.mat');

net1=dagnn.DagNN.loadobj(net);
net1.mode = 'test' ;
net1.conserveMemory=false;

net2=dagnn.DagNN.loadobj(net);
net2.mode = 'test' ;
net2.conserveMemory=false;
IM_NUM=21;

for i=1:IM_NUM
    path1 = ['./IV_images/IR',num2str(i),'.png'];
    path2 = ['./IV_images/VIS',num2str(i),'.png'];
    fuse_path = ['./fused_infrared_X/fused',num2str(i),'_propose.png'];
    fuse_path_weight = ['./fused_infrare_weight/fused',num2str(i),'_propose.png'];
    
    %path1=['./noise_images/image1_gau_005_left.png'];
    %path2=['./noise_images/image1_gau_005_right.png'];conv1
    %varnames={'conv1xxx','res2cx','res3b7x','res4b35x','res5cx';2,4,8,16,31};
    varnames={'conv1','res2a_branch1','res3b7x','res4b35x','res5cx';2,4,8,16,31};
    [img1,img2]=path2double_img(path1,path2);
    
    
    % Highpass filter test image
    npd = 16;
    fltlmbd = 5;
    [I_lrr1, I_saliency1] = lowpass(img1, fltlmbd, npd);
    [I_lrr2, I_saliency2] = lowpass(img2, fltlmbd, npd);
    %% fuison lrr parts
    F_lrr = (I_lrr1+I_lrr2)/2;
    
    [saliency_a,saliency_b]=make_3c_single(I_saliency1,I_saliency2);
    net1.eval({'data', saliency_a}) ;
    net2.eval({'data', saliency_b}) ;
    %% -----0
    out_0_a=net1.vars(net1.getVarIndex('data')).value;
    out_0_b=net2.vars(net2.getVarIndex('data')).value;
    unit_0=1;
    l1_featrues_0_a = extract_l1_feature(out_0_a);
    l1_featrues_0_b = extract_l1_feature(out_0_b);
    [F_saliency_0, l1_featrues_0_ave_a, l1_featrues_0_ave_b,weight_0_a,weight_0_b] = ...
            my_fusion_strategy(l1_featrues_0_a, l1_featrues_0_b, I_saliency1, I_saliency2, unit_0);
    
    %% conv1xxx²ã----1
    out_1_a=net1.vars(net1.getVarIndex(varnames{1,1})).value;
    out_1_b=net2.vars(net2.getVarIndex(varnames{1,1})).value;
    unit_1=varnames{2,1};
    l1_featrues_1_a = extract_l1_feature(out_1_a);
    l1_featrues_1_b = extract_l1_feature(out_1_b);
    [F_saliency_1, l1_featrues_1_ave_a, l1_featrues_1_ave_b,weight_1_a,weight_1_b] = ...
            my_fusion_strategy(l1_featrues_1_a, l1_featrues_1_b, I_saliency1, I_saliency2, unit_1);
    %% res2cx²ã----2
    out_2_a=net1.vars(net1.getVarIndex(varnames{1,2})).value;
    out_2_b=net2.vars(net2.getVarIndex(varnames{1,2})).value;
    unit_2=varnames{2,2};
    l1_featrues_2_a = extract_l1_feature(out_2_a);
    l1_featrues_2_b = extract_l1_feature(out_2_b);
    [F_saliency_2, l1_featrues_2_ave_a, l1_featrues_2_ave_b,weight_2_a,weight_2_b] = ...
            my_fusion_strategy(l1_featrues_2_a, l1_featrues_2_b, I_saliency1, I_saliency2, unit_2);
    %% res3b7x²ã----3
    out_3_a=net1.vars(net1.getVarIndex(varnames{1,3})).value;
    out_3_b=net2.vars(net2.getVarIndex(varnames{1,3})).value;
    unit_3=varnames{2,3};
    l1_featrues_3_a = extract_l1_feature(out_3_a);
    l1_featrues_3_b = extract_l1_feature(out_3_b);
    [F_saliency_3, l1_featrues_3_ave_a, l1_featrues_3_ave_b,weight_3_a,weight_3_b] = ...
            my_fusion_strategy(l1_featrues_3_a, l1_featrues_3_b, I_saliency1, I_saliency2, unit_3);
    %% res4b35x²ã----4
    out_4_a=net1.vars(net1.getVarIndex(varnames{1,4})).value;
    out_4_b=net2.vars(net2.getVarIndex(varnames{1,4})).value;
    unit_4=varnames{2,4};
    l1_featrues_4_a = extract_l1_feature(out_4_a);
    l1_featrues_4_b = extract_l1_feature(out_4_b);
    [F_saliency_4, l1_featrues_4_ave_a, l1_featrues_4_ave_b,weight_4_a,weight_4_b] = ...
            my_fusion_strategy(l1_featrues_4_a, l1_featrues_4_b, I_saliency1, I_saliency2, unit_4);
    %% res5cx----5
    out_5_a=net1.vars(net1.getVarIndex(varnames{1,5})).value;
    out_5_b=net2.vars(net2.getVarIndex(varnames{1,5})).value;
    unit_5=varnames{2,5};
    l1_featrues_5_a = extract_l1_feature(out_5_a);
    l1_featrues_5_b = extract_l1_feature(out_5_b);
    [F_saliency_5, l1_featrues_5_ave_a, l1_featrues_5_ave_b,weight_5_a,weight_5_b] = ...
            my_fusion_strategy(l1_featrues_5_a, l1_featrues_5_b, I_saliency1, I_saliency2, unit_5);
    %% 
    %F_saliency = max(F_saliency_1, F_saliency_2);
    %F_saliency = max(F_saliency, F_saliency_3);
    %F_saliency = max(F_saliency, F_saliency_4);
    %F_saliency = max(F_saliency, F_saliency_5);
    F_weight_a=extend_max_six(weight_1_a,weight_2_a,weight_3_a,weight_4_a,weight_5_a,weight_0_a);
    F_weight_b=extend_max_six(weight_1_b,weight_2_b,weight_3_b,weight_4_b,weight_5_b,weight_0_b);
    F_saliency_w=I_saliency1.*F_weight_a+I_saliency2.*F_weight_b;
    
    F_saliency=extend_max_six(F_saliency_1,F_saliency_2,F_saliency_3,F_saliency_4,F_saliency_5,F_saliency_0);
    
    fusion_im = F_lrr + F_saliency;
    fusion_im_w=F_lrr+F_saliency_w;
    imwrite(fusion_im,fuse_path,'png');
    imwrite(fusion_im_w,fuse_path_weight,'png');
end

function max_value=extend_max_six(v1,v2,v3,v4,v5,v6)
    max_value=max(v1,v2);
    max_value=max(max_value,v3);
    max_value=max(max_value,v4);
    max_value=max(max_value,v5);
    max_value=max(max_value,v6);
end
function max_value=extend_max_five(v1,v2,v3,v4,v5)
    max_value=max(v1,v2);
    max_value=max(max_value,v3);
    max_value=max(max_value,v4);
    max_value=max(max_value,v5);
end

function [img1,img2]=path2double_img(path1,path2)
    img1=imread(path1);
    img2=imread(path2);
    img1=im2double(img1);
    img2=im2double(img2);
end

function [img3c_1,img3c_2]=make_3c_single(image1,image2)
    image1=make_3c(image1);
    image2=make_3c(image2);
    img3c_1=single(image1);
    img3c_2=single(image2);
end



function var_5layers=fetch_layer_feature(net,s)
    var_5layers=struct(s(1).varnames,net.vars(net.getVarIndex(s(1).varnames)).value,...
                       s(2).varnames,net.vars(net.getVarIndex(s(2).varnames)).value,...
                       s(3).varnames,net.vars(net.getVarIndex(s(3).varnames)).value,...
                       s(4).varnames,net.vars(net.getVarIndex(s(4).varnames)).value,...
                       s(5).varnames,net.vars(net.getVarIndex(s(5).varnames)).value);
end