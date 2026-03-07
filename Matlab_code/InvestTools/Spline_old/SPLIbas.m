function [B,x,k,breaks]=SPLIBAS(n,a,b,x,order,k,breaks);
% SPLIBAS computes polynomial spline basis.
%
% INPUTS:
%   n - the number of basis functions
%   a,b - lower and upper endpoints
%   x   - k-vector of the evaluation points 
%          (default: k-1 knot averages)
%   order - the order of differentiation (default: 0)
%           if a vector, SPLIBAS returns a cell array 
%           otherwise it returns a matrix
% Auxillary inputs:
%   k: the order of the spline (pieces are k-1 polynomials)
%       (default: 3)
%   breaks: user specified breakpoint sequence
%       (default: evenly spaced single breakpoints)
%
% Consistency requires that length(breaks)=n-k+1
%
% OUTPUTS
%   B - a kxn basis matrix
%   x,k,breaks : these are returned (useful if defaults values 
%        must be computed) 
%
% Uses SPLIDOP
%
% See also: SPLINODE, SPLIAUX, SPLIDOP, FUNBAS, FUNNODE, FUNEVAL

% Copyright (c) 1997 by Paul L. Fackler

  if nargin<3, error('At least three parameters must be passed'); end
  if nargin<4, x=[]; end
  if nargin<5 | isempty(order), order=0; end
  if nargin<6, k=[]; end
  if nargin<7, breaks=[]; end
  
  % GET DEFAULTS
  if isempty(k) | isempty(breaks)
    [k,breaks]=SPLIAUX(n,a,b,k,breaks);
  end
  
  if isempty(x)
    x=SPLINODE(n,a,b,k,breaks);
  end
  
  % A FEW CHECKS
  if k<1
    error(['Incorrect value for spline order (k): ' num2str(k)]);
  end
  if min(size(breaks))>1
    error('''breaks'' must be a vector');
  end
  if length(breaks)~=n-k+1
    error('length(breaks) must equal n-k+1');
  end
  if order>k
    error('Order of differentiation cannot be greater than k');
  end
  if size(x,2)>1
    error('x must be a column vector')
  end
  
  p=length(breaks);
  m=size(x,1); 
  minorder=min(order);

  % Augment the breakpoint sequence 
  augbreaks=[a(ones(k-minorder,1));breaks(:);b(ones(k-minorder,1))];
  
  % The following lines determine the maximum index of 
  %   the breakpoints that are less than or equal to x,
  %   (if x=b use the index of the next to last breakpoint).
%  [temp,ind]=sort([-inf;breaks(2:end-1);x(:)]);
%  temp=find(ind>=p);
%  j=ind(temp)-(p-1);
%  ind=temp-(1:m)';
%  ind(j)=ind(:)+(k-minorder);    % add k-minorder for augmented sequence
  ind=lookup(augbreaks,x,3);
  
  % Recursively determine the values of a k-order basis matrix.
  % This is placed in an (m x k+1-order) matrix
  bas=zeros(m,k-minorder+1);
  bas(:,1)=ones(m,1);
  B=cell(length(order),1);
  if max(order)>0, D=SPLIDOP(n,a,b,max(order),k,breaks); end % Derivative op
  if minorder<0, I=SPLIDOP(n,a,b,minorder,k,breaks); end    % Integral op
  for j=1:k-minorder
    for jj=j:-1:1
      b0=augbreaks(ind+jj-j);          
      b1=augbreaks(ind+jj);
      temp=bas(:,jj)./(b1-b0);
      bas(:,jj+1)=(x-b0).*temp+bas(:,jj+1);
      bas(:,jj)=(b1-x).*temp;
    end
    % as now contains the order j spline basis
    ii=find((k-j)==order);
    if ~isempty(ii)
      ii=ii(1);
      % Put values in appropriate columns of a sparse matrix
      r=(1:m)'; r=r(:,ones(k-order(ii)+1,1));
      c=(order(ii)-k:0)-(order(ii)-minorder); 
      c=c(ones(m,1),:)+ind(:,ones(k-order(ii)+1,1));
      B{ii}=sparse(r,c,bas(:,1:k-order(ii)+1),m,n-order(ii));
      % If needed compute derivative or anti-derivative operator
      if order(ii)>0
        B{ii}=B{ii}*D{order(ii)};
      elseif order(ii)<0
        B{ii}=B{ii}*I{-order(ii)};
      end
      %B{ii}=full(B{ii});
    end
  end
  
  if length(order)==1, B=B{1}; end