function [middle, labels, forward_label, adjw, normlpmtxw, region] = supervoxel_generation_syn(im, labels)
%CBCT_FATURES
num_bins = 120;
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

num_labels = max(max(max(labels)));
labels= labels(minx:maxx,miny:maxy,minz:maxz);

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

middle = middle(a,:);
middle(:,1) = middle(:,1) + minx-1;
middle(:,2) = middle(:,2) + miny-1;
middle(:,3) = middle(:,3) + minz-1;

f_his = f_his(a,:);

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