function [x,k,breaks]=splinode(n,a,b,k,breaks)
% SPLINODE Computes standard nodes for splines using knot averaging.
% USAGE:
%   x=SPLINODE(n,a,b,k,breaks)
%
% See Also: SPLIBAS, SPLIAUX, FUNBAS, FUNEVAL.

% Copyright (c) 1997 by Paul L. Fackler

if nargin<3 error('3 parameters must be specified'); end
if nargin<4, k=[]; end
if nargin<5, breaks=[]; end

if isempty(k) | isempty(breaks)
  [k,breaks]=SPLIAUX(n,a,b,k,breaks);
end

x=cumsum([a*ones(k,1);breaks(:);b*ones(k,1)]);
x=(x(1+k:n+k)-x(1:n))/k;