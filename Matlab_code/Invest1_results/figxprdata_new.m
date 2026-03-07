% Resulting output (4 states and 1 control: stochastic)
% Policy when 3 states are given (Relationship with the fourth one)
clear all;
load su77521							% Loading workspace of run model
sfile = 'figxprdata_new';

t = 0;
if t==T
   c = [];
else
   c = copt(:,t+1);
end

Lt = 600;
Wt = 500000;
P = [Pmin:5:Pmax];
R = [330,365,430]';
%_________________________________________________
% Policy Function x(P) for different states gievn

S = cell(1,length(R));
XP = zeros(length(P),length(R));
      for ri = 1:length(R)
         Rt = R(ri);
         S{ri} = zeros(length(P),length(n));
         S{ri}(:,1) = Rt;
         S{ri}(:,2) = P;
         S{ri}(:,3) = Lt;
         S{ri}(:,4) = Wt;
         XP(:,ri) = feval(vmaxh,S{ri},c,t);
      end
save(sfile);