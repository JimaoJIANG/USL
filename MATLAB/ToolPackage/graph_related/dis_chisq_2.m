function D = dis_chisq_2( X, Y )   
%%
% edition2
% dis = 1 - 2 * sum((xi - yi)^2 / (xi + yi))
m = size(X,1);  n = size(Y,1);  
mOnes = ones(1,m); D = zeros(m,n);  
for i=1:n  
  yi = Y(i,:);  yiRep = yi( mOnes, : );% yiRep = yi repeat m times,m*p matrix  
  s = yiRep + X;    d = yiRep - X;    
  D(:,i) = sum(d.^2 ./ (s+eps),2);  
end  
 mnOnes = ones(m,n);  
 D = mnOnes - 2*D;  