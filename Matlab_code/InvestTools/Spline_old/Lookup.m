function ind=lookup(tabvals,x,endadj);
% LOOKUP  Performs a table lookup.
% USAGE:
%   ind=lookup(tabvals,x,endadj);
% INPUTS:
%   tabvals: a sorted vector of values
%   x: a matrix a values
%   endadj: a optional endpoint adjustment: 0, 1, 2 or 3.
% Returns a matrix of size(x) with element (i,j) equal to
%   max k: x(i,j)>=tabvals(k)
%
% Optional endpoint adjustments:
%   0: no adjustments
%   1: values of x < min(tabvals) will return 
%        length(tabvals=tabvals(1))
%   2: values of x > max(tabvals) will return 
%        m-length(tabvals=tabvals(end))
%   3: adjustments 1 and 2 will be performed

% Copyright (c) 1997 by Paul L. Fackler

if nargin<2
  error('At least two parameters must be specified');
end
if nargin<3 endadj=0; end
if isempty(endadj) endad=0; end

n=prod(size(x));
if min(size(tabvals))>1
  error('tabvals must be a vector');
else 
  tabvals=tabvals(:);
  if any(diff(tabvals)<0)
    error('tabvals must be sorted in ascending order')
  end
end
m=length(tabvals);
if endadj>=2, m=m-length(find(tabvals==tabvals(end))); end

[temp,ind]=sort([tabvals(1:m); x(:)]);
temp=find(ind>m);
j=ind(temp)-m;
ind=reshape(temp-(1:n)',size(x));
ind(j)=ind(:);

if endadj==1 | endadj==3
  ind(ind==0)=length(find(tabvals==tabvals(1))); 
end
