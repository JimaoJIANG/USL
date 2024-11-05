function D = dis_chisq_1( X, Y )  
%% ø®∑Ωæ‡¿Î edition 1
% dis = 0.5 * sum((xi - yi) ^2 / ( xi + yi))
%%% supposedly it's possible to implement this without a loop!  
%
m = size(X,1);  n = size(Y,1);  
mOnes = ones(1,m); D = zeros(m,n);  
for i=1:n  
  yi = Y(i,:);  yiRep = yi( mOnes, : );  
  s = yiRep + X;    d = yiRep - X;  
  D(:,i) = sum( d.^2 ./ (s+eps), 2 );  
end  
D = D/2; 
%}
%D = sum(X .* Y);
