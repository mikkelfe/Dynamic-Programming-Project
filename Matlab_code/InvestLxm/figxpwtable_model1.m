clear all;

load figxpwdatath1_model1

WW = (ones(size(P)))'*W';
PP = P'*(ones(size(W)))';
PPs = (1-tcs).*PP + (1-fds).*fme;
PPps = (1+tcb).*PP + fme;
negind = find(XP<0);
PPps(negind) = (1-tcs).*PP(negind) + (1-fds).*fme;

LAFA = (WW./(PPs.*Lt))-1;
LAFA12f = (WW - PPs.*Lt - PPps.*XP)./(PPs.*(Lt+XP));

[WW(99,:)' LAFA(99,:)' XP(99,:)' LAFA12f(99,:)']
[WW(154,:)' LAFA(154,:)' XP(154,:)' LAFA12f(154,:)']