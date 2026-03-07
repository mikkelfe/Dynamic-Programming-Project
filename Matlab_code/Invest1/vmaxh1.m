function [x,v] = vmaxh1(s,c,t);
% Solves Bellman equation at state nodes: stochastic problem 
% Uses hybrid method:
% Calculates optimal v and x for each set of state nodes

global bound
                              % compute bounds of x for s: xl,xu are nn by 1
							         % rows of s =rows of xl=rows of xu
[xl,xu] = feval(bound,s);	
nn = size(s,1);
[vxq xq] = vx(s,c,t,xl,xu);	% vxq is (q+2 by nn), xq is (q+2 by nn)
[v ind] = max(vxq);				% v is (1 by nn), ind is (1 by nn)
x = zeros(1,nn);			      % x is (1 by nn)
for j=1:nn
   x(j) = xq(ind(j),j);
end 
v = v';						      % now v is (nn by 1)
x = x';								% now x is (nn by 1)
