function output=gaussFilter(I,sigma)

output = I.*0; 
window=double(uint8(3*sigma)*2+1);%���ڴ�Сһ��Ϊ3*sigma  
H=fspecial('gaussian', window, sigma);%fspecial('gaussian', hsize, sigma)�����˲�ģ��  
%Ϊ�˲����ֺڱߣ�ʹ�ò���'replicate'������ͼ����ⲿ�߽�ͨ�������ڲ��߽��ֵ����չ��

for c=1:size(I,3)
    output(:,:,c)=imfilter(I(:,:,c),H,'replicate');
end

end