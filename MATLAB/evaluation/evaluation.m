clear;
addpath(genpath('..\ToolPackage'))

img_dir = '\path\to\your\data\';
para_dir = '\path\to\your\data\';
seg_dir = '\path\to\your\data\';
test_dir = '\path\to\your\data\';

ref_img_path = '\path\to\your\data\';
ref_para_path = '\path\to\your\data\';
ref_seg_path = '\path\to\your\data\';
ref_test_path = '\path\to\your\data\';

label_transfer_dir = '\path\to\your\output\dir\';
basenum = 48;

files = dir([img_dir, '*.mha']);
num =length(files);
numlabels = 9;
numlabelsbi = 7;
oupt = [1 1; 2 2;3 4; 5 5; 6 6; 7 7; 8 9];

tacc = zeros(num,numlabels);
tdsc = zeros(num,numlabels);
tahd = zeros(num,numlabels);

ite = 1;
for file_i = 1:num
    tempname = files(file_i).name;
    filename_test = [img_dir, tempname];
    
    ref = mha_read_volume(ref_img_path);
    testim = mha_read_volume(filename_test);
    
    [r,c,d] =size(testim);

    para1 = load(ref_para_path);
    para2 = load([para_dir, strrep(tempname,'mha', 'mat')]);
    seg1 = mha_read_volume(ref_seg_path);
    seg2 = mha_read_volume([seg_dir, tempname]);

   % combine left and right
    for ii = 1:7
        tsyn1 = seg1 == oupt(ii,1);
        seg1 = seg1 + (oupt(ii,2) - oupt(ii,1)) * uint16(tsyn1);
        
        tsyn1 = seg2 == oupt(ii,1);
        seg2 = seg2 + (oupt(ii,2) - oupt(ii,1)) * uint16(tsyn1);
    end
    
    t1 = load([ref_test_path]);
    t2 = load([test_dir, '\test_', strrep(tempname,'mha','mat')]);
    

    base1 = t1.re_base;
    base2 = t2.re_base;
    C = t2.C;
    C_opt = C;
    t1 = double(t1.re_fea);
    t2 = double(t2.re_fea);

    f1 = para1.is_forward;
    m1 = double(para1.newmid) + 1;
    opt1 = full(para1.lpmtxw);
    adj1 = full(para1.sparse_adj);
    l1 = para1.labels;

    f2 = para2.is_forward;
    m2 = double(para2.newmid) + 1;
    opt2 = full(para2.lpmtxw);
    adj2 = full(para2.sparse_adj);
    l2 = para2.labels;
   
    if isfield(para1, 'region') && isfield(para2, 'region')
        region1 = para1.region;
        region2 = para2.region;
        templabel = ones(size(ref));
        templabel(region1(1):region1(2),region1(3):region1(4),region1(5):region1(6)) = l1;
        l1 = templabel;
        l1(l1 == 0) = l1(l1 == 0) + 1;
        templabel = ones(size(testim));
        templabel(region2(1):region2(2),region2(3):region2(4),region2(5):region2(6)) = l2;
        l2 = templabel;
        l2(l2 == 0) = l2(l2 == 0) + 1;
    end
    
    base1 = base1(1:size(m1,1),:);
    base2 = base2(1:size(m2,1),:);
    t1 = t1(1:size(m1,1),:);
    t2 = t2(1:size(m2,1),:);

    fac1 = base1(:,1:basenum)' * t1;
    fac2 = base2(:,1:basenum)' * t2;

    P1 = C * base1';
    P2 = pdist2(base2,P1');
    P2 = abs(P2);
    P2 = normalize(P2, 'norm',1);
    [t4,t5] = min(P2,[],2);

    num1 = size(t1,1);
    num2 = size(t2,1);
    segnum1 = zeros(num1,20);
    segnum2 = zeros(num2,20);
    segmax1 = zeros(num1,1);
    segmax2 = zeros(num2,1);
    invind1 = zeros(110000,1);
    invind2 = zeros(110000,1);
    supnum1 = zeros(num1,1);
    supnum2 = zeros(num2,1);
    seg3 = zeros(size(testim),'uint16');
    for i = 1 : num1
        invind1(f1(i)) = i;
    end
    for i = 1 : num2
        invind2(f2(i)) = i;
    end
    [r1, c1, d1]= size(ref);
    [r2, c2, d2]= size(testim);
    r3 = min(r1, r2);
    c3 = min(c1, c2);
    d3 = min(d1, d2);

    for i = 1 : r3
        for j = 1 : c3
            for k = 1: d3
                if invind2(l2(i,j,k)) ~= 0 
                    tempid = m1(t5(invind2(l2(i,j,k))),:);
                    seg3(i,j,k) = seg1(tempid(1),tempid(2),tempid(3));
                end
                if invind1(l1(i,j,k)) ~= 0 
                    supnum1(invind1(l1(i,j,k))) = supnum1(invind1(l1(i,j,k))) + 1;
                    if seg1(i,j,k)~=0
                        segnum1(invind1(l1(i,j,k)),seg1(i,j,k)) = segnum1(invind1(l1(i,j,k)),seg1(i,j,k)) + 1;
                    end
                end
                if invind2(l2(i,j,k)) ~= 0 
                    supnum2(invind2(l2(i,j,k))) = supnum2(invind2(l2(i,j,k))) + 1;
                    if seg2(i,j,k)~=0
                        segnum2(invind2(l2(i,j,k)),seg2(i,j,k)) = segnum2(invind2(l2(i,j,k)),seg2(i,j,k)) + 1;
                    end
                end
                %}
            end
        end
    end
    %
    for i = 1 : num1
        if sum(segnum1(i,:))  > 1
            [t4,segmax1(i)] = max(segnum1(i,:));
        end
    end
    for i = 1 : num2
        if sum(segnum2(i,:))  > 1
            [t4,segmax2(i)] = max(segnum2(i,:));
        end
    end

    total = zeros(numlabels, 1);
    segmax3 = zeros(num2,1);
    for i = 1:num2
        if segmax1(t5(i)) > 0
            total(segmax1(t5(i))) = total(segmax1(t5(i))) +1;
            segmax3(i) = segmax1(t5(i));
        end
    end
    
    for i = 1:numlabels
        ts1 = segnum1(:,i) >= 1;
        ts2 = segnum2(:,i) >= 1;
        ts3 = zeros(num2,1);
        for j = 1 : num2
            if ts1(t5(j)) 
                ts3(j) = 1;
            end
        end
        ts4 = 2*sum(double(ts2) .* double(ts3));
        ts5 = sum(double(ts2) + double(ts3));
        tdsc(ite,i) = double(ts4) / double(ts5);
    end
    ite = ite + 1;
    
    % label transfer
    seg_res_volume = zeros(r2,c2,d2);
    for i = 1:r2
        for j = 1:c2
            for k = 1:d2
                tempp1 = find(f2 == l2(i,j,k));
                if ~isempty(tempp1)
                    tempp2 = t5(tempp1);
                    if segmax1(tempp2)>0 && testim(i, j, k) > 1100
                        seg_res_volume(i,j,k) =segmax1(tempp2);
                    end
                end
            end
        end
    end
    
    for i = 2:r2-1
        for j = 2:c2-1
            for k = 2:d2-1
                if seg_res_volume(i,j,k) == 0 && testim(i,j,k) > 1000
                    templabel = zeros(9,1);
                    for tti = -1:1
                        for ttj = -1:1
                            for ttk = -1:1
                                if seg_res_volume(i+tti,j+ttj,k+ttk) > 0
                                    templabel(seg_res_volume(i+tti,j+ttj,k+ttk)) = templabel(seg_res_volume(i+tti,j+ttj,k+ttk)) + 1;
                                end
                            end
                        end
                    end
                    if sum(templabel) > 12
                        [~,seg_res_volume(i,j,k)] = max(templabel);
                    end
                end

            end
        end
    end

    writemetaimagefilevf([label_transfer_dir, tempname],seg_res_volume,[1,1,1],[r/2,-c/2,d/2]);
end

%%
tdscmean = mean(tdsc(1:num,:));
tdscstd = std(tdsc(1:num,:));
dscmean = zeros(1,numlabelsbi);
dscstd = zeros(1,numlabelsbi);
for k = 1:numlabelsbi
    dscmean(k) = tdscmean(oupt(k,2));
    dscstd(k) = tdscstd(oupt(k,2));
end
close all;
