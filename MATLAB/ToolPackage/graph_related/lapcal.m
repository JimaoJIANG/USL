function [lap,w] = lapcal(fea,adj)
%LAPCAL
[r,c] = size(fea);
dis = zeros(r,r);
w = zeros(r,r);
for i = 1:r
    for j = 1:r
        if adj(i,j) == 1
            dis(i,j) = norm(fea(i,:) - fea(j,:));
            %dis(i,j) = (fea(i,:) - fea(j,:)) * (fea(i,:) - fea(j,:))';
            if isnan(dis(i,j)) 
                a=1;
            end
            dis(j,i) = dis(i,j);
        end
    end
end
[~,~,delta] = find(dis);
delta = median(delta);
delta = median(median(delta));
for i = 1:r
    for j = 1:r
        if adj(i,j) == 1
            w(i,j) = exp((-(dis(i,j) )^2) / (2 * delta^2));
            w(j,i) = w(i,j);
        end
    end
end
tempw = sum(w);
lap = -w + diag(tempw);
end

