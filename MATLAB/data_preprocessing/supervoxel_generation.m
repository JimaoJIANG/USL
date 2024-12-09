function [features, middle, labels, forward_label, adjw, normlpmtxw,region] = supervoxel_generation(im, numreqiredsupervoxels, compactness)
%CBCT_FATURES
pattern = [randi([-120,120],50,3); randi([-100,100],100,3); randi([-50,50],100,3);randi([-20,20],100,3)];
num_bins = 120;
num_spacon = 100;
min_gray_scale = 0;
max_gray_scale = 4500;
supervoxel_threshold = 0.5;

temp = sum(im,[2,3]);
temp = find(temp>0);
minx = temp(1);
maxx = temp(end);

temp = sum(im,[1,3]);
temp = find(temp>0);
miny = temp(1);
maxy = temp(end);

temp = sum(im,[1,2]);
temp = find(temp>0);
minz = temp(1);
maxz = temp(end);

region = [minx,maxx,miny,maxy,minz,maxz];
cutted_im = im(minx:maxx,miny:maxy,minz:maxz);

stack = single(cutted_im);
stack(cutted_im > max_gray_scale) = max_gray_scale;
stack = 128 * (stack - min(stack, [], 'all')) / (max(stack, [], 'all') - min(stack, [], 'all')); 
stack = uint8(stack);
[labels, num_labels] = slicsupervoxelmex(stack, numreqiredsupervoxels, compactness);


[r,c,d] = size(cutted_im);
if ~exist('mask','var')
    tmp1 = cutted_im > min_gray_scale;
    tmp2 = cutted_im < max_gray_scale;
    mask = double(tmp1 & tmp2);
end

ground = zeros(num_labels,1);
num_of_each_supervoxel = zeros(num_labels,1);
middle = zeros(num_labels,3);
        
for i = 1:r
    for j = 1:c
        for k = 1:d
            if labels(i,j,k) <= 0
                continue;
            end
            middle(labels(i,j,k),:) = middle(labels(i,j,k),:) + [i j k];
            num_of_each_supervoxel(labels(i,j,k)) = num_of_each_supervoxel(labels(i,j,k)) + 1;
            ground(labels(i,j,k)) = ground(labels(i,j,k)) + mask(i,j,k);
        end
    end
end

middle = round(middle ./ num_of_each_supervoxel);
average_ground = ground ./ num_of_each_supervoxel;
ground = average_ground > supervoxel_threshold;
a = find(ground);

num_remained_supervoxels = size(a,1);
adj = adj_sparse( labels , double(num_labels));
f_his = supervox_his(cutted_im, labels, num_labels, num_bins, min_gray_scale, max_gray_scale) + 1;

num_chisq = size(pattern,1);
f_chisq = zeros(num_labels,num_chisq);
for i = 1:num_labels
    if ground(i) == 0
        continue;
    end
    t1 = f_his(i,:);
    t2 = middle(i,:);
    for j = 1:num_chisq
        t31 = t2(1) + pattern(j,1);
        t32 = t2(2) + pattern(j,2);
        t33 = t2(3) + pattern(j,3);
        if(t31 < 1)
            t31 = 1;
        end
        if(t31 > r)
            t31 = r;
        end
        if(t32 < 1)
            t32 = 1;
        end
        if(t32 >c)
            t32 =c;
        end
        if(t33 < 1)
            t33 = 1;
        end
        if(t33 > d)
            t33 = d;
        end
        t3 = labels(t31,t32,t33);
        if t3 == 0
            continue;
        end
        f_chisq(i,j) = dis_chisq(t1,f_his(t3,:));
    end
    
end

middle = middle(a,:);
middle(:,1) = middle(:,1) + minx-1;
middle(:,2) = middle(:,2) + miny-1;
middle(:,3) = middle(:,3) + minz-1;

f_his = f_his(a,:);
f_chisq = f_chisq(a,:);

f_his = f_his / max(max(f_his));
f_chisq = f_chisq / max(max(f_chisq));

features = [f_his * 5 f_chisq];

f_sta = zeros(num_remained_supervoxels, num_spacon);
fea_sta = zeros(num_remained_supervoxels, num_spacon);
for i = 1 : num_remained_supervoxels
    for j = 1 : num_remained_supervoxels
        fea_sta(i,j) = norm(middle(i,:) - middle(j,:));
    end
    fea_sta(i,:) = sort(fea_sta(i,:));
    f_sta(i,:) = resample(fea_sta(i,1 : num_remained_supervoxels), 100, num_remained_supervoxels);
end
f_sta = f_sta / max(max(f_sta));

features = [features f_sta * 10];

forward_label = a(a~=0);
adj = adj(forward_label,forward_label);
adj = double(adj);
adj = adj - speye(size(adj,1));

[lpmtxw,adjw] = lapcal(f_his,adj);
lpmtxw = sparse(lpmtxw);
adjw = sparse(adjw);

adjw = adjw + speye(num_remained_supervoxels);
dadj = sum(adjw);
dadj = dadj .^ (-0.5);
adjw = diag(dadj) * adjw * diag(dadj);
normlpmtxw = diag(dadj) * lpmtxw * diag(dadj);
end