clear;
addpath(genpath('..\ToolPackage'));
img_dir = '\path\to\your\data\';
para_dir = '\path\to\your\data\';
output_dir = '\path\to\your\output\dir';

files = dir([img_dir, '*.mha']);
num =length(files);
for k = 1:num
    file_name = files(k).name;
    fprintf('%d, %s\n', k, file_name);
    refimg = mha_read_volume([img_dir, file_name]);
    refimg = double(refimg);
    [r, c ,d] = size(refimg);

    reflabels = zeros(r, c, d,'uint16');
    reftemp = load([para_dir, strrep(file_name,'mha', 'mat')]);
    region1 = reftemp.region;
    reflabels(region1(1):region1(2),region1(3):region1(4),region1(5):region1(6)) = reftemp.labels;
    refforward = reftemp.is_forward;
    mkdir([output_dir, strrep(file_name,'.mha', '')]);
    mkdir([output_dir, strrep(file_name,'.mha', ''), '\img']);
    mkdir([output_dir, strrep(file_name,'.mha', ''), '\para']);
    for i = 1:30
            spacing=[16 16 16];
            O_grid=make_init_grid(spacing,[r, c, d]);
            X_d = 16*(rand(size(O_grid))-0.5);
            O_trans=reshape(X_d,size(O_grid))+O_grid; 
            [trans_volume,trans_field] = bspline_transform(O_trans,refimg,spacing,3);
            [x,y,z] = ndgrid(1:r, 1:c, 1:d);
            x_prime = x + trans_field(:,:,:,1);
            y_prime = y + trans_field(:,:,:,2);
            z_prime = z + trans_field(:,:,:,3);
            trans_volume = int16(trans_volume(1:r,1:c,1:d));
            writemetaimagefilevf([output_dir, strrep(file_name,'.mha', ''), '\img\', num2str(i),'.mha'],trans_volume,[1,1,1],[r/2,-c/2,d/2]);

            trans_labels = interpn(x,y,z,reflabels,x_prime,y_prime,z_prime,'nearest');
            trans_labels = trans_labels(1:r, 1:c, 1:d);
            [middle, labels, is_forward, adjw, normlpmtxw, region] = supervoxel_generation_syn(trans_volume, trans_labels);
            middle = middle - 1;
            
            templabel = zeros(r, c, d);
            templabel(region(1):region(2),region(3):region(4),region(5):region(6)) = labels;
            newmid = middle + 1;
            for j = 1:size(newmid,1)
                if templabel(newmid(j,1),newmid(j,2),newmid(j,3)) ~= is_forward(j)
                    ttt = find(templabel == is_forward(j));
                    [ttx,tty,ttz] = ind2sub([r, c, d],ttt);
                    tempinx = randi(size(ttx,1));
                    newmid(j,:) = [ttx(tempinx),tty(tempinx),ttz(tempinx)];
                end
            end
            newmid = newmid - 1;
            
            middle = uint16(middle);
            labels = uint16(labels);
            is_forward = uint16(is_forward);
            path = [output_dir, strrep(file_name,'.mha', ''), '\para\', num2str(i),'.mat'];
            save(path, 'is_forward', 'newmid', 'middle', 'labels', 'adjw', 'normlpmtxw','region');
    end
end