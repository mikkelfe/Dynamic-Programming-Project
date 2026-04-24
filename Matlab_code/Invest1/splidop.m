function D=splidop(n,a,b,order,k,breaks)
% splidop  Creates differential operator matrices for polynomial splines.
% USEAGE:
%   d=splidop(n,a,b,order,k,breaks);  
% Computes the matrix operator that maps a vector of spline
% coefficients into the vector of coefficients of the derivative.
% Example:
%    g(x)=B^k(x)*c;
%    g'(x)=B^{k-1}(x)*BSDOP(n,a,b,1)*c;
%    g"(x)=B^{k-1}(x)*BSDOP(n,a,b,2)*c;
% Integrals are computed with ORDER<1 (anti-derivatives).
%    int_a^x g(x)=B^{k+1}(x)*BSDOP(n,a,b,-1)
%
% D is returned as a cell array of sparse matrices w/ abs(order) 
%   elements; to view it use full(D{i})
%
% Most users will not need this function; 
%   use SPLIBAS or FUNEVAL instead.
%
% See also SPLIBAS, FUNEVAL.

% Copyright (c) 1997 by Paul L. Fackler

  if nargin<3 error('At least three parameters must be specified'), end
  if nargin<4 | isempty(order); order=1; end
  if nargin<5, k=[]; end
  if nargin<6, breaks=[]; end
  
  if isempty(k) | isempty(breaks)
    [k,breaks]=spliaux(n,a,b,k,breaks);
  end

  if n-k<=0 error('Too small a value of n'); end
  if order>k
    error('Order of derivative operator cannot be larger than k');
  end
  
  kk=max(k-1,k-order-1);
  augbreaks=[a*ones(kk,1);breaks(:);b*ones(kk,1)];  
  
  D=cell(abs(order),1);
  if order>0                             % derivative
    temp=k./(augbreaks(k+1:n+k-1)-augbreaks(1:n-1));
    D{1}=spdiags([-temp temp],0:1,n-1,n);
    for i=2:order
      temp=(k+1-i)./(augbreaks(k+1:n+k-i)-augbreaks(i:n-1));
      D{i}=spdiags([-temp temp],0:1,n-i,n+1-i)*D{i-1};
    end
  elseif order<0                         % anti-derivative (integral)
    temp=(augbreaks(kk+2:kk+n+1)-augbreaks(kk-k+1:kk+n-k))/(k+1);
    D{1}=sparse(tril(ones(n+1,1)*temp',-1));
    for i=-2:-1:order
      temp=(augbreaks(kk+2:kk+n-i)-augbreaks(kk-k+i+2:kk+n-k))/(k-i);
      D{-i}=sparse(tril(ones(n-i,1)*temp',-1))*D{-1-i};
    end
  end