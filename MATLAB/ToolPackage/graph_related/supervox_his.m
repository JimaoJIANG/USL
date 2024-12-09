function [ supervoxhis ] = supervox_his( im, labels, numlabels, numbins, mmin, mmax)
%SUPERVOX_HIS

bins =numbins;
supervoxhis = zeros(numlabels,bins);
[r,c,d] = size(im);

Max = mmax;
Min = mmin;
mm = double(Max - Min);
delta = (mm +1)/ bins;

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

