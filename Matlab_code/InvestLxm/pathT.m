clear all;
load errors3

%load su77521Lxm
load su77521Lxmth1

sfile2 = 'pathT';

R = 365;
P = 1775;
L = 600;
W = 1000000;
syear = 0;
%syear = [0,10,20,30];

years = 10;
s0 = [R P L W];
gT = cell(1,4);            
gT{1} = zeros(years,4);        %for storing only means for all years
%gT{2} = zeros(years,4);
%gT{3} = zeros(years,4);
%gT{4} = zeros(years,4);

pout = cell(1,4);              
pout{1} = zeros(years,6);        %for storing only means for all years
%pout{2} = zeros(years,6);
%pout{3} = zeros(years,6);
%pout{4} = zeros(years,6);

%%mutual fund
XM = zeros(years,2);


trials = size(E123{1},1);        % each e is trials 500 by 2 for 2 states
sj = s0;

for tind = 1:length(syear)
yri = syear(tind);
%yri = 0;
sjt = zeros(trials,4);
gjt = zeros(size(sjt));
sjt(:,1) = sj(1);
sjt(:,2) = sj(2);
sjt(:,3) = sj(3);
sjt(:,4) = sj(4);
%%%
xmtr = zeros(trials,1);
%%%

for yr = yri:(yri+years-1)              %0:T (0:19)
   if yr==T
      c1 = [];
   else
      c1 = copt(:,yr+1);
   end
   ini =  find(sjt(:,3)>=(sminv(3)-1) & sjt(:,4)>0);  %note L=[0,400:2000],sminv(3)=400
   outi = find(sjt(:,3)<1 | sjt(:,4)<=0);           %Besides bankruptcy, for W=0 also, V is known
   sk = sjt(ini,:);
   %xk = feval(vmaxh,sk,c1,yr);                         %%%%
   [xk1 xk2 vk] = feval('vmaxh',sk,c1,yr);
   xk = [xk1 xk2];
   gk = zeros(size(sk));
   eyr = E123{yr-yri+1};
   eyr1 = eyr(ini,:);
   gk = gstate(sk,xk,eyr1);
   gjt(ini,:) = gk;
   if isempty(outi)==0
      gjt(outi,3) = 0;
      s4o = sjt(outi,4);
      ro = ones(size(s4o,1),1).*(EgRM-1);
      oi = find(s4o<0);
      ro(oi) = rn;
      gjt(outi,4) = (1+ro).*s4o;
   end
   gT{tind}(yr+1-yri,3:4) = mean(gjt(:,3:4),1);                             %%%%
   gT{tind}(yr+1-yri,1:2) = var(gjt(:,3:4),1);
   %%%%
   xmtr(ini) = xk2;
   %%%%
   outbr0i = find(gjt(:,3)<1 | gjt(:,4)<=0);
   inibr0i =  find(gjt(:,3)>=(sminv(3)-1) & gjt(:,4)>0); 
   outbri = find(gjt(:,4)<=0);
   pout{tind}(yr+1-yri,3) = length(outbr0i)./500;       %prob of out of farming = BR+choosing out
   pout{tind}(yr+1-yri,1) = length(outbri)./500;     %prob of BR
   pout{tind}(yr+1-yri,2) = (length(outbr0i) - length(outbri))./500;  %prob of choosing out
   
   gjt2v = (1-tcs).*gjt(inibr0i,2) + (1-fds).*fme;
   if isempty(inibr0i)==0
   outdau = find((gjt(inibr0i,4) - (1-dau).*gjt2v.*gjt(inibr0i,3))<=0);
   pout{tind}(yr+1-yri,4) = length(outdau)./length(inibr0i);
   pout{tind}(yr+1-yri,5) = mean((gjt(inibr0i,4)./(gjt2v.*gjt(inibr0i,3))))-1;
   pout{tind}(yr+1-yri,6) = mean(gjt(inibr0i,3));
   %%%%
   XM(yr+1-yri,1) = mean(xmtr(inibr0i),1);
   XM(yr+1-yri,2) = mean((xmtr(inibr0i).*exp(gamma0 + eyr(inibr0i,3))),1);
   %%%%
   end
   sjt = gjt;
end
end
save(sfile2);