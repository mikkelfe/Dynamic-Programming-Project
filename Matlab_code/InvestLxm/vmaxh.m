function [x,xm,v] = vmaxh(s,c,t)
% Solves Bellman equation at state nodes: stochastic problem 
% Uses hybrid method:
% Calculates optimal v and x for each set of state nodes

                              % compute bounds of x for s: xl,xu are nn by 1
							         % rows of s =rows of xl=rows of xu
                              
[xl,xu] = feval('boundl',s);
nn = size(s,1);
[vxq xq xmq] = vx(s,c,t,xl,xu);	% vxq is (q+2 by nn), xq is (q+2 by nn)
[v ind] = max(vxq);				% v is (1 by nn), ind is (1 by nn)
x = zeros(1,nn);			      % x is (1 by nn)
xm = zeros(1,nn);			      % xm is (1 by nn)
for j=1:nn
   x(j) = xq(ind(j),j);
   xm(j) = xmq(ind(j),j);
end 
v = v';						      % now v is (nn by 1)
x = x';								% now x is (nn by 1)
xm = xm';							% now xm is (nn by 1)