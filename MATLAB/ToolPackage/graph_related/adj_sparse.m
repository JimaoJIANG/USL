function adj = adj_sparse(labels , numlabels)
adj = zeros(numlabels,'int8');
[r,c,d] = size(labels);
for i = 2:r-1
    for j = 2:c-1
        for k = 2:d-1
            if labels (i,j,k) * labels(i+1,j,k) *labels(i,j+1,k)*labels(i,j,k+1)* labels(i-1,j,k) *labels(i,j-1,k)*labels(i,j,k-1)==0
                continue;
            end
            adj(labels (i,j,k) , labels(i+1,j,k)) = 1;
            adj(labels (i,j,k) , labels(i,j+1,k)) = 1;
            adj(labels (i,j,k) , labels(i,j,k+1)) = 1;
            adj(labels (i,j,k) , labels(i-1,j,k)) = 1;
            adj(labels (i,j,k) , labels(i,j-1,k)) = 1;
            adj(labels (i,j,k) , labels(i,j,k-1)) = 1;
        end
    end
end
end

