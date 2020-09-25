function [gen, ave1, ave2,weight1,weight2] = my_fusion_strategy(features_a, features_b, source_a, source_b, unit)

[m,n] = size(features_a);
[m1,n1] = size(source_a);
ave_temp1 = zeros(m1,n1);
ave_temp2 = zeros(m1,n1);
weight_ave_temp1 = zeros(m1,n1);
weight_ave_temp2 = zeros(m1,n1);
w=1/16*[1 2 1;2 4 2;1 2 1];
E1=conv2(features_a.^2,w,'same');
E2=conv2(features_b.^2,w,'same');

%for i=2:m-1
%    for j=2:n-1
for i=1:m
    for j=1:n
        %A1 =sum(sum(features_a(i-1:i+1,j-1:j+1)))/9;
        %A2 =sum(sum(features_b(i-1:i+1,j-1:j+1)))/9;
        %A1 =sum(sum(features_a(i:i+1,j:j+1)))/4;
        %A2 =sum(sum(features_b(i:i+1,j:j+1)))/4;
        % weight average
        %weight_ave_temp1(((i-2)*unit+1):((i-1)*unit),((j-2)*unit+1):((j-1)*unit)) = A1/(A1+A2);
        %weight_ave_temp2(((i-2)*unit+1):((i-1)*unit),((j-2)*unit+1):((j-1)*unit)) = A2/(A1+A2);
        %ave_temp1(((i-2)*unit+1):((i-1)*unit),((j-2)*unit+1):((j-1)*unit)) = A1;
        %ave_temp2(((i-1)*unit+1):((i-1)*unit),((j-1)*unit+1):((j-1)*unit)) = A2;
        
        
        ave_temp1(((i-1)*unit+1):((i)*unit),((j-1)*unit+1):((j)*unit)) = E1(i,j);
        ave_temp2(((i-1)*unit+1):((i)*unit),((j-1)*unit+1):((j)*unit)) = E2(i,j);
    end
end

%weight_ave_temp1 = weight_ave_temp1(1:m1,1:n1);
%weight_ave_temp2 = weight_ave_temp2(1:m1,1:n1);
weight_ave_temp1 = ave_temp1(1:m1,1:n1)./(ave_temp1(1:m1,1:n1)+ave_temp2(1:m1,1:n1));
weight_ave_temp2 = ave_temp2(1:m1,1:n1)./(ave_temp1(1:m1,1:n1)+ave_temp2(1:m1,1:n1));
gen = source_a.*weight_ave_temp1 + source_b.*weight_ave_temp2;
ave1 = ave_temp1;
ave2 = ave_temp2;
weight1=weight_ave_temp1;
weight2=weight_ave_temp2;
end