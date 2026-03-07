function [xmqi,vqi] = vmaxhm(s,c,t,xqi)
% Solves Bellman equation at state nodes: stochastic problem 
% Uses hybrid method:
% Calculates optimal v and x for each set of state nodes

                              % compute bounds of x for s: xl,xu are nn by 1
							         % rows of s =rows of xl=rows of xu
[xml,xmu] = feval('boundm',s,xqi);
nn = size(s,1);
[vxq2 xq2] = vxm(s,c,t,xqi,xml,xmu);	% vxq is (q+2 by nn), xq is (q+2 by nn)
[vqi ind] = max(vxq2);				% vqi is (1 by nn), ind is (1 by nn)
xmqi = zeros(1,nn);			      % xmqi is (1 by nn)
for j=1:nn
   xmqi(j) = xq2(ind(j),j);
end 
vqi = vqi';						      % now v is (nn by 1)
xmqi = xmqi';							% now xm is (nn by 1)