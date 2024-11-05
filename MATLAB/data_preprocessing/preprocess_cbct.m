clear;
addpath(genpath('.\SLIC_mex'));
addpath(genpath('..\ToolPackage'));
file_dir = '\path\to\your\data\';
output_dir = '\path\to\your\output\dir\';

files = dir([file_dir, '*.mha']);
num =length(files);
for i = 1:num
    file_name = files(i).name;
    fprintf('%d, %s\n', i, file_name);
    temp = [file_dir, file_name];
    output_file_name = strrep(file_name,'mha','mat');
    output_path = [output_dir, output_file_name];
    
    img =  mha_read_volume(temp);
    [fea, middle, labels, is_forward, adjw, normlpmtxw, region] = supervoxel_generation(img, 10000, 10.0);
    middle = middle - 1;
    
    [r, c ,d] = size(img);
    templabel = zeros(r, c, d);
    templabel(region(1):region(2), region(3):region(4), region(5):region(6)) = labels;
    newmid = middle + 1;
    for j = 1:size(newmid,1)
        if templabel(newmid(j,1), newmid(j,2), newmid(j,3)) ~= is_forward(j)
            ttt = find(templabel == is_forward(j));
            [ttx, tty, ttz] = ind2sub([r, c, d], ttt);
            tempinx = randi(size(ttx, 1));
            newmid(j, :) = [ttx(tempinx), tty(tempinx), ttz(tempinx)];
        end
    end
    newmid = newmid - 1;
    
    fea = single(fea);
    middle = uint16(middle);
    labels = uint16(labels);
    is_forward = uint16(is_forward);
    save(output_path,'fea', 'is_forward', 'middle', 'newmid', 'labels', 'adjw', 'normlpmtxw', 'region');
end
