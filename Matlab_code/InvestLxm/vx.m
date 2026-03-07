function [vxq,xq,xqm] = vx(s,c,t,xl,xu);
% For each node, it computes v as a function of x when we plug state equation 
% into value function and then plug a specific state s (the state node)
% vxq is (q+2 by nn), xq is (q+2 by nn)

global q

nn= size(s,1);							% rows of s =rows of xl=rows of xu
xq = zeros(q+2,nn);
xqm = zeros(q+2,nn);
vxq = zeros(q+2,nn);
rxlu = xu-xl;									% Range from xl to xu
gap = rxlu ./(q-1);
xqi = zeros(nn,1);
vxqi = zeros(nn,1);

for qi=1:(q+2)
   if qi==1
      xqi = zeros(nn,1);
   elseif qi==2
      xqi = -s(:,3);
   else
      xqi = xl+(gap.*(qi-3));
   end
   [xmqi,vqi] = vmaxhm(s,c,t,xqi);
   xq(qi,:)=xqi';
   xqm(qi,:) = xmqi';
   vxq(qi,:) = vqi';
end