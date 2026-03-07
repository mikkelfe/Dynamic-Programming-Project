function [vxq,xq] = vxi(s,c,t,xl,xu);
% For each node, it computes v as a function of x when we plug state equation 
% into value function and then plug a specific state s (the state node)
% vxq is (q by nn), xq is (q by nn)


global q valuef bound

nn= size(s,1);							% rows of s =rows of xl=rows of xu
xq = zeros(q,nn);
vxq = zeros(q,nn);
rxlu = xu-xl;									% Range from xl to xu
gap = rxlu ./(q-1);
xqi = zeros(nn,1);
vxqi = zeros(nn,1);
for qi=1:q
   xqi = xl+(gap.*(qi-1));
   vxqi = feval(valuef,s,c,t,xqi); % xqi is nn by 1, s is nn by 4
   xq(qi,:)=xqi';
   vxq(qi,:) = vxqi';
end
