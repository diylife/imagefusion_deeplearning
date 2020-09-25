%% Li H, Wu X J. DenseFuse: A Fusion Approach to Infrared and Visible Images[J]. arXiv preprint arXiv:1804.08361, 2018.
%% https://arxiv.org/abs/1804.08361
cbf_f_path=['F:\matlabPro\imagefusion_deeplearning\compared methods\CBF_Image_Fusion_Codes\fused\'];
convsr_f_path=["F:\matlabPro\imagefusion_deeplearning\compared methods\ConvSR_Image_Fusion_Codes\CSR_Fusion_Author\fused\"];
jsr_f_path=['F:\matlabPro\imagefusion_deeplearning\compared methods\JSR_JSRSD_Image_Fusion_Codes\fused_jsr\'];
jsrsd_f_path=['F:\matlabPro\imagefusion_deeplearning\compared methods\JSR_JSRSD_Image_Fusion_Codes\fused_jsrsd\'];
wls_f_path=['F:\matlabPro\imagefusion_deeplearning\compared methods\WLS_Image_Fusion_Codes\fused\'];
propose_f_path=['F:\matlabPro\imagefusion_deeplearning\fused_infrare_weight\'];

a=[cbf_f_path,convsr_f_path,jsr_f_path,jsrsd_f_path,wls_f_path,propose_f_path];
b=["cbf","convsr","jsr","jsrsd","wls","propose"];
fid=fopen('results.txt','w');
IM_NUM=21;
for j=1:6
    fprintf(fid,'\n');
    fprintf(fid,'%s-------------------------\n',b(j));
    disp(strcat('��ʼ',b(j)))
    tEN=0;tMI=0;tQabf=0;tFMI_pixel=0;tFMI_dct=0;tFMI_w=0;tNabf=0;tSCD=0;tSSIM=0;tMS_SSIM=0;
    for i=1:IM_NUM
        source_image1 = imread(['F:\matlabPro\imagefusion_deeplearning\IV_images\IR',num2str(i),'.png']);
        source_image2 = imread(['F:\matlabPro\imagefusion_deeplearning\IV_images\VIS',num2str(i),'.png']);
        files=dir(char(a(j)));
        suffix=strsplit(files(length(files)).name,"_");
        image_path=[char(a(j)),'fused',num2str(i),'_',suffix{1,2}];
        
        fused=imread(image_path);
        fprintf('������%d��\t',i);
        disp(image_path);
        [EN,MI,Qabf,FMI_pixel,FMI_dct,FMI_w,Nabf,SCD,SSIM, MS_SSIM] = analysis_Reference(fused,source_image1,source_image2);
        fprintf(fid,'%2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f\r\n',EN,MI,Qabf,FMI_pixel,FMI_dct,FMI_w,Nabf,SCD,SSIM, MS_SSIM);
        disp([EN,MI,Qabf,FMI_pixel,FMI_dct,FMI_w,Nabf,SCD,SSIM, MS_SSIM]);
        fprintf('��%d�ŷ������\n',i);
        tEN=tEN+EN;
        tMI=tMI+MI;
        tQabf=tQabf+Qabf;
        tFMI_pixel=tFMI_pixel+FMI_pixel;
        tFMI_dct=tFMI_dct+FMI_dct;
        tFMI_w=tFMI_w+FMI_w;
        tNabf=tNabf+Nabf;
        tSCD=tSCD+SCD;
        tSSIM=tSSIM+SSIM;
        tMS_SSIM=tMS_SSIM+MS_SSIM;
    end
    fprintf(fid,'%s\n','��ֵ');
    fprintf(fid,'%2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f\r\n',tEN/IM_NUM,tMI/IM_NUM,tQabf/IM_NUM,tFMI_pixel/IM_NUM,tFMI_dct/IM_NUM,tFMI_w/IM_NUM,tNabf/IM_NUM,tSCD/IM_NUM,tSSIM/IM_NUM, tMS_SSIM/IM_NUM);
    disp(strcat(b(j),'����'))
end
fclose(fid);

