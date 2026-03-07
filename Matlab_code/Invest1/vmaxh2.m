function [x,v] = vmaxh2(s,c,t);
% Solves Bellman equation at state nodes: stochastic problem 
% Uses hybrid method:
% Calculates optimal v and x for each set of state nodes

global bound q
                              % compute bounds of x for s: xl,xu are nn by 1
							         % rows of s =rows of xl=rows of xu
[xl,xu] = feval(bound,s);
nn = size(s,1);
[vxq xq] = vxi(s,c,t,xl,xu);				% vxq is (q+2 by nn), xq is (q+2 by nn)
[v ind] = max(vxq);				% v is (1 by nn), ind is (1 by nn)
xopt = zeros(1,nn);			% xopt is (1 by nn)
for j=1:nn
   xopt(j) = xq(ind(j),j);
end 
x = xopt';								% x is (nn by 1)

clear vxq xq ind
%**********************************

xas = 1600/(q-1);                % Lmax-Lmin = 1600
xlf = max(xl,(x - xas));
xuf = min(xu,(x + xas));

[vxq xq] = vxf(s,c,t,xlf,xuf);				% vxq is (q+2 by nn), xq is (q+2 by nn)
[v ind] = max(vxq);				% v is (1 by nn), ind is (1 by nn)
xopt = zeros(1,nn);			% xopt is (1 by nn)
for j=1:nn
   xopt(j) = xq(ind(j),j);
end 
v = v';						      % now v is (nn by 1)
x = xopt';								% x is (nn by 1)
