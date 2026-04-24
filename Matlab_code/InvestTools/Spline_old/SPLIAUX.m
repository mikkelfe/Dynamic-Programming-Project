function [k,breaks]=spliaux(n,a,b,k,breaks);
%SPLIAUX Defines default auxilliary parmaeters for spline functions
%
%USAGE: [k,breaks]=SPLIAUX(n,a,b,k,breaks);
%
% Default order (k) is 3 (cubic) if breaks is unspecified; 
%   otherwise it equals n-length(breaks)
% Default breaks are simple and evenly spaced on [a,b].
%
% See Also:  SPLIBAS, FUNEVAL, FUNBAS

% Copyright (c) 1997, by Paul L. Fackler

if nargin<3
  error('3 parameters must be passed');
end

if nargin<4 | isempty(k)
  if nargin<5 | isempty(breaks)
    k=3;                         % default is cubic splines
  else
    k=n-length(breaks)+1;
  end
end

if prod(size(k,1))>1 
  error(['Spline order (k) has improper size']);
end

if k<1
  error(['Spline order (k) is too small: ' num2str(k)]);
end

if nargin<5 | isempty(breaks)
  breaks=(a:(b-a)/(n-k):b)';    % default is evenly spaced breakpoints
end  

if any(breaks<a) | any(breaks>b)
  error('Breakpoints must be within the (a,b) interval');
end

if any(diff(breaks))<0
  error('Breakpoints must be non-decreasing');
end

if breaks(2)==a | breaks(end-1)==b
  error('Breakpoint multiplicities are not allowed at endpoints')
end

if breaks(1)~=a | breaks(end)~=b
  error('Breakpoints must include interval endpoints (a and b)')
end

if length(breaks)~=n-k+1
  errstr={'Incompatible dimensions in n, k and breaks';...
     'length(breaks)=n-k+1';
    ['length(breaks): ' num2str(length(breaks))];
    ['             n: ' num2str(n)];
    ['             k: ' num2str(k)]};
  disp(errstr)
  error(' ');
end
