function [ supervoxhis ] = supervox_his( im, labels, numlabels, numbins, mmin, mmax)
%SUPERVOX_HIS 此处显示有关此函数的摘要
%   此处显示详细说明
% im = int32(im);
bins =numbins;
supervoxhis = zeros(numlabels,bins);
[r,c,d] = size(im);

Max = mmax;
Min = mmin;
mm = double(Max - Min);
delta = (mm +1)/ bins;
%MM = delta * bins;

%pim = im;
%pim = im - Min;
%{
pim = double(pim>Min) .* double(pim);
pim = double(pim);
pim = (pim < Max) .* pim;
%}
for i = 1:r
    for j = 1:c
        for k = 1:d
            tempp = im(i,j,k);
            if tempp > Max
                tempp = Max;
                %continue;
            end
            if tempp < Min
                tempp = Min;
                % continue;
            end
            t = floor((tempp - Min) / delta) + 1;
            if t > numbins
                t = numbins;
            end
            if t < 1
                t = 1;
            end
            if labels(i,j,k)==0
                continue;
            end
            supervoxhis(labels(i,j,k),t) = supervoxhis(labels(i,j,k),t) + 1;
        end
    end
end

end

