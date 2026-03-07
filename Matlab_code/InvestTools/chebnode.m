function x=chebnode(n,a,b)
% CHEBNODE Computes standard nodes for Chebyshev polynomials
% USAGE:
%   x=chebnode(n,a,b)
% Evaluates the roots of the order n Chebyshev polynomial and transforms
%   them to the interval [a,b].
%
% CHEBNODE_opts can be set using OPTSET:
%   0 : usual nodes
%   1 : nodes extended to endpoints
%   2 : Lobatto nodes
%
% See Also: CHEBBAS, FUNNODE.

% Copyright (c) 1997, 1999 by Paul L. Fackler & Mario J. Miranda


global CHEBNODE_opts;

if isempty(CHEBNODE_opts) 
  CHEBNODE_opts.nodetype=0;
  if nargin==0, return; end
end

s=(b-a)/2;
m=(b+a)/2;

if CHEBNODE_opts.nodetype<2                           % usual nodes 
  k=pi*(0.5:(max(n)-0.5))';
  x=m(1)-cos(k(1:n(1))/n(1))*s(1);
  if CHEBNODE_opts.nodetype==1                        % Extend nodes to endpoints
    aa=x(1);
    bb=x(end);
    x=(bb*a-aa*b)/(bb-aa)+(b-a)/(bb-aa)*x;
  end
else                                                  % Lobatto nodes
  k=pi*(0:(max(n)-1))';
  x=m(1)-cos(k(1:n(1))/(n(1)-1))*s(1);
end