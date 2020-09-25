%% Li H, Wu X J. DenseFuse: A Fusion Approach to Infrared and Visible Images[J]. arXiv preprint arXiv:1804.08361, 2018.
%% https://arxiv.org/abs/1804.08361
%gtf_f_path=['F:\matlabPro\imagefusion_deeplearning\compared methods\GTF\'];
dtcwt_f_path=["F:\matlabPro\imagefusion_deeplearning\compared methods\DTCWT\"];


a=[dtcwt_f_path];
b=["dtcwt"];
fid=fopen('result_add_a.txt','w');
IM_NUM=21;
for j=1:1
    fprintf(fid,'\n');
    fprintf(fid,'%s-------------------------\n',b(j));
    disp(strcat('开始',b(j)))
    tEN=0;tMI=0;tQabf=0;tFMI_pixel=0;tFMI_dct=0;tFMI_w=0;tNabf=0;tSCD=0;tSSIM=0;tMS_SSIM=0;
    for i=1:IM_NUM
        source_image1 = imread(['F:\matlabPro\imagefusion_deeplearning\IV_images\IR',num2str(i),'.png']);
        source_image2 = imread(['F:\matlabPro\imagefusion_deeplearning\IV_images\VIS',num2str(i),'.png']);
        files=dir(char(a(j)));
        suffix=strsplit(files(length(files)).name,"_");
        image_path=[char(a(j)),'fused',num2str(i),'_',suffix{1,2}];
        
        fused=imread(image_path);

        if i==3
            fused=imresize(fused,[510,505]);
        end
        if i==7
            fused=imresize(fused,[328,595]);
        end
        
        if i==11
            fused=imresize(fused,[247,359]);
        end
        
        if i==19
            fused=imresize(fused,[475,575]);
        end
        
        fprintf('分析第%d张\t',i);
        disp(image_path);
        [EN,MI,Qabf,FMI_pixel,FMI_dct,FMI_w,Nabf,SCD,SSIM, MS_SSIM] = analysis_Reference(fused,source_image1,source_image2);
        fprintf(fid,'%2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f\r\n',EN,MI,Qabf,FMI_pixel,FMI_dct,FMI_w,Nabf,SCD,SSIM, MS_SSIM);
        disp([EN,MI,Qabf,FMI_pixel,FMI_dct,FMI_w,Nabf,SCD,SSIM, MS_SSIM]);
        fprintf('第%d张分析完成\n',i);
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
    fprintf(fid,'%s\n','均值');
    fprintf(fid,'%2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f\r\n',tEN/IM_NUM,tMI/IM_NUM,tQabf/IM_NUM,tFMI_pixel/IM_NUM,tFMI_dct/IM_NUM,tFMI_w/IM_NUM,tNabf/IM_NUM,tSCD/IM_NUM,tSSIM/IM_NUM, tMS_SSIM/IM_NUM);
    disp(strcat(b(j),'结束'))
end
fclose(fid);

