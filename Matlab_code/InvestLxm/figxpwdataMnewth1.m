% Resulting output (4 states and 1 control: stochastic)
% Policy when 3 states are given (Relationship with the fourth one)
clear all;

load su77521Lxmth1
%load su77521Lxm							% Loading workspace of run model
sfile = 'figxpwdataMnewth1';

t = 0;
if t==T
   c = [];
else
   c = copt(:,t+1);
end

Lt = 600;
Rt = 365;
P = [Pmin:5:Pmax];
W = [400000:200000:4000000]';
%_________________________________________________
% Policy Function x(P) for different states gievn

S = cell(1,length(W));
XP1 = zeros(length(P),length(W));
XP2 = zeros(length(P),length(W));
      for wi = 1:length(W)
         Wt = W(wi);
         S{wi} = zeros(length(P),length(n));
         S{wi}(:,1) = Rt;
         S{wi}(:,2) = P;
         S{wi}(:,3) = Lt;
         S{wi}(:,4) = Wt;
         [XP1(:,wi) XP2(:,wi)] = feval('vmaxh',S{wi},c,t);
      end
save(sfile);