function xgrid = cgrid(xcoord,d)

% CARTGRID: xgrid = cgrid(xcoord)
%
% Forms grid in R^d from Cartesian product of coordinate vectors
%
% On input, xcoord by a d by 1 cell array whose ith entry is
% is an n(i) by 1 vector of coordinates in dimension i. On
% output, xgrid is the prod(n) by d vector of grid points
% formed by the Cartesian product of the coordinates vectors. 

if nargin==1
   d = length(xcoord);
   n = zeros(d,1);
   for i=1:d
      n(i) = length(xcoord{i});
   end
   p = 1; q = prod(n);
   xgrid = zeros(q,d);
   for i=1:d
      q = q/n(i);
      xi = xcoord{i}(:,ones(1,q))';
      xi = xi(:);
      xi = xi(:,ones(1,p));
      xgrid(:,i) = xi(:);
      p = p*n(i);
   end   
else
   n = length(xcoord);
   p = 1; q = n^d;
   xgrid = zeros(q,d);
   for i=1:d
      q = q/n;
      xi = xcoord(:,ones(1,q))';
      xi = xi(:);
      xi = xi(:,ones(1,p));
      xgrid(:,i) = xi(:);
      p = p*n;
   end   
end
