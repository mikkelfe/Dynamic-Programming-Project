% Resulting output (4 states and 1 control: stochastic)
% Policy when 3 states are given (Relationship with the fourth one)
clear all;
load su77521theta0var44					% Loading workspace of run model
sfile = 'figxpwdatath0var44';

t = 0;
if t==T
   c = [];
else
   c = copt(:,t+1);
end

Lt = 600;
Rt = 365;
P = [Pmin:5:Pmax];
W = [400000:150000:1600000]';
%_________________________________________________
% Policy Function x(P) for different states gievn

S = cell(1,length(W));
XP = zeros(length(P),length(W));
      for wi = 1:length(W)
         Wt = W(wi);
         S{wi} = zeros(length(P),length(n));
         S{wi}(:,1) = Rt;
         S{wi}(:,2) = P;
         S{wi}(:,3) = Lt;
         S{wi}(:,4) = Wt;
         XP(:,wi) = feval(vmaxh,S{wi},c,t);
      end
save(sfile);