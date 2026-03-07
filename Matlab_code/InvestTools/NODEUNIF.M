function [x,xcoord] = nodeunif(n,a,b)

% NODEUNIF: [x,xcoord] = nodeunif(n,a,b)
%
% Computes uniform nodes for intervals R^n
%
% If dimension d of n, a, and b is one, returns n by
% by 1 vector x containing n uniform nodes spanning
% the interval [a,b]. If dimension d>1, returns d by 1
% cell array xcoord whose kth entry is the n(k) by 1
% vector of n(k) uniform nodes spanning the interval
% [a(k),b(k)]; also returns prod(n) by d vector x of
% grid points created by forming Cartesian product of
% vectors in xcoord.

d = length(n);
if d==1
   x = linspace(a,b,n)';
else
   for k=1:d
      xcoord{k} = linspace(a(k),b(k),n(k))';
   end   
   x = cgrid(xcoord);
end
