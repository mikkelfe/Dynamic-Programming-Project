function [vxq2,xmq2] = vxm(s,c,t,xqi,xml,xmu)

% For each node, it computes v as a function of x when we plug state equation 
% into value function and then plug a specific state s (the state node)
% vxq is (q+2 by nn), xq is (q+2 by nn)

global q2

nn= size(s,1);							% rows of s =rows of xl=rows of xu
xmq2 = zeros(q2,nn);
vxq2 = zeros(q2,nn);
rxlu = xmu-xml;									% Range from xl to xu
gap = rxlu ./(q2-1);
xmqj = zeros(nn,1);
vxqj = zeros(nn,1);
for qj=1:q2
   xmqj = xml+(gap.*(qj-1));
   x = [xqi xmqj];
   vxqj = feval('valuefw',s,c,t,x); %  is nn by 1, s is nn by 4
   xmq2(qj,:)=xmqj';
   vxq2(qj,:) = vxqj';
end