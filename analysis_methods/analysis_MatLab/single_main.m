
%% Li H, Wu X J. DenseFuse: A Fusion Approach to Infrared and Visible Images[J]. arXiv preprint arXiv:1804.08361, 2018.
%% https://arxiv.org/abs/1804.08361
cbf_f_path=['F:\matlabPro\imagefusion_deeplearning\compared methods\CBF_Image_Fusion_Codes\fused\'];
convsr_f_path=["F:\matlabPro\imagefusion_deeplearning\compared methods\ConvSR_Image_Fusion_Codes\CSR_Fusion_Author\fused\"];
jsr_f_path=['F:\matlabPro\imagefusion_deeplearning\compared methods\JSR_JSRSD_Image_Fusion_Codes\fused_jsr\'];
jsrsd_f_path=['F:\matlabPro\imagefusion_deeplearning\compared methods\JSR_JSRSD_Image_Fusion_Codes\fused_jsrsd\'];
wls_f_path=['F:\matlabPro\imagefusion_deeplearning\compared methods\WLS_Image_Fusion_Codes\fused\'];
propose_f_path=['F:\matlabPro\imagefusion_deeplearning\fused_infrare_weight\'];


IM_NUM=1
for i=1:IM_NUM
    source_image1 = imread(['F:\matlabPro\imagefusion_deeplearning\save\IR',num2str(i),'.jpg']);
    source_image2 = imread(['F:\matlabPro\imagefusion_deeplearning\save\VIS',num2str(i),'.jpg']);
%     files=dir(char(a(j)));
%     suffix=strsplit(files(length(files)).name,"_");
%     image_path=[char(a(j)),'fused',num2str(i),'_',suffix{1,2}];

    fused=imread(['F:\matlabPro\imagefusion_deeplearning\save\fused_image',num2str(i),'.jpg']);
    fprintf('分析第%d张\t',i);
    [EN,MI,Qabf,FMI_pixel,FMI_dct,FMI_w,Nabf,SCD,SSIM, MS_SSIM] = analysis_Reference(fused,source_image1,source_image2);
%     fprintf(fid,'%2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f\r\n',EN,MI,Qabf,FMI_pixel,FMI_dct,FMI_w,Nabf,SCD,SSIM, MS_SSIM);
    disp([EN,MI,Qabf,FMI_pixel,FMI_dct,FMI_w,Nabf,SCD,SSIM, MS_SSIM]);
    fprintf('第%d张分析完成\n',i);
end


