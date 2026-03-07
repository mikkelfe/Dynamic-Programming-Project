clear all;
load errors

load su77521


sfile2 = 'figpathdistdata_W600';

R = 365;
P = 1775;
L = 600;
W = 600000;
syear = 0;
%syear = [0,10,20,30];

years = 10;
%years = length(E);            % E = [e1,e2,...,e(T+1)]'
s0 = [R P L W];
gs = cell(1,3);               %storing for selected t distributions: t=1,5,10
glafa = cell(1,3);
gT = cell(1,1);             
gT{1} = zeros(years,4);        %for storing only means for all years
%gT{2} = zeros(years,4);
%gT{3} = zeros(years,4);
%gT{4} = zeros(years,4);
%gT{5} = zeros(years,4);
%gT{6} = zeros(years,4);
%gT{7} = zeros(years,4);
pout = cell(1,1);               %storing for selected t distributions: t=1,5,10,20
pout{1} = zeros(years,6);        %for storing only means for all years
%pout{2} = zeros(years,6);
%pout{3} = zeros(years,6);
%pout{4} = zeros(years,6);
%pout{5} = zeros(years,6);
%pout{6} = zeros(years,6);
%pout{7} = zeros(years,6);

trials = size(E{1},1);        % each e is trials 500 by 2 for 2 states
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

for yr = yri:(yri+years-1)              %0:T (0:19)
   if yr==T
      c1 = [];
   else
      c1 = copt(:,yr+1);
   end
   ini =  find(sjt(:,3)>=(sminv(3)-1) & sjt(:,4)>0);  %note L=[0,400:2000],sminv(3)=400
   outi = find(sjt(:,3)<1 | sjt(:,4)<=0);           %Besides bankruptcy, for W=0 also, V is known
   sk = sjt(ini,:);
   xk = feval(vmaxh,sk,c1,yr);                         %%%%
   gk = zeros(size(sk));
   eyr = E{yr-yri+1};
   eyr1 = eyr(ini,:);
   gk = gstate(sk,xk,eyr1);
   gjt(ini,:) = gk;
   if isempty(outi)==0
      gjt(outi,3) = 0;
      s4o = sjt(outi,4);
      ro = ones(size(s4o,1),1).*rp;
      oi = find(s4o<0);
      ro(oi) = rn;
      gjt(outi,4) = (1+ro).*s4o;
   end
   gT{tind}(yr+1-yri,3:4) = mean(gjt(:,3:4),1);                             %%%%
   gT{tind}(yr+1-yri,1:2) = var(gjt(:,3:4),1);     
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
   end
   sjt = gjt;
   if yr==0
      gs{1} = gjt;         %for years 1,5,10
      glafa{1} = (gjt(inibr0i,4)./(gjt2v.*gjt(inibr0i,3)))-1;
   elseif yr == 4
      gs{2} = gjt;
      glafa{2} = (gjt(inibr0i,4)./(gjt2v.*gjt(inibr0i,3)))-1;
   elseif yr == 9
      gs{3} = gjt;
      glafa{3} = (gjt(inibr0i,4)./(gjt2v.*gjt(inibr0i,3)))-1;
   end
end
end
save(sfile2);