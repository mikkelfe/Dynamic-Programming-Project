function [B,x]=chebbas(n,a,b,x,order);
% CHEBBAS Computes basis matrices for Chebyshev polynomials
% SYNTAX:
%   x=chebbas(n,a,b,x,order);
% INPUTS:
%   n       : the number of basis functions (1 plus the polynomial order)
%   a       : the left endpoint
%   b       : the right endpoint
%   x       : k-vector of the evaluation points 
%             (default: roots of order n Chebyshev polynomial)
%   order   : the order of differentiation (default: 0)
%             if a vector, SPLIBAS returns a cell array 
%             otherwise it returns a matrix
% OUTPUTS:
%   B :  a kxn basis matrix or cell array of basis matrices
%   x :  evaluation points (useful if defaults values are computed)
%
% See Also: CHEBNODE, CHEBDOP, CHEBBAS, FUNBAS.

% Copyright (c) 1997, 1999 by Paul L. Fackler & Mario J. Miranda

  if nargin<3, error('3 parameters must be specified'), end
  if nargin<4, x=[]; end
  if nargin<5 | isempty(order), order=0; end

  minorder=min(0,min(order));
  
  if isempty(x) & optget('CHEBNODE','nodetype')==0      % evaluate at standard nodes     
     x=chebnode(n,a,b);
     temp=((n-0.5):-1:0.5)';
     bas=cos((pi/n)*temp*(0:(n-1-minorder)));
  else
    if isempty(x)                        % evaluate at standard nodes     
      x=chebnode(n,a,b);
    end
    z = (2/(b-a))*(x-(a+b)/2);
    m=size(z,1);
    bas=zeros(m,n-minorder);
    bas(:,1)=ones(m,1);
    bas(:,2)=z;
    z=2*z;
    for i=3:n-order
      bas(:,i)=z.*bas(:,i-1)-bas(:,i-2);
    end
  end
    
  if length(order)==1
    if order~=0
      D=chebdop(n,a,b,order);
      B=bas*D{abs(order)};
      %B=bas*D{1};
    else
      B=bas;
    end
  else
    B=cell(length(order),1);
    maxorder=max(order);
    if maxorder>0, D=chebdop(n,a,b,maxorder); end
    if minorder<0, I=chebdop(n,a,b,minorder); end
    for ii=1:length(order)
      if order(ii)==0
        B{ii}=bas(:,1:n);
      elseif order(ii)>0
        B{ii}=bas(:,1:n)*D{order(ii)};
      else
        B{ii}=bas(:,1:n-order(ii))*I{-order(ii)};
      end
    end
  end
