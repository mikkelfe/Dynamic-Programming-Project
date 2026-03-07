clear all;

load figxpwdataMnewth1

WW = (ones(size(P)))'*W';
PP = P'*(ones(size(W)))';
PPs = (1-tcs).*PP + (1-fds).*fme;
PPps = (1+tcb).*PP + fme;
negind = find(XP1<0);
PPps(negind) = (1-tcs).*PP(negind) + (1-fds).*fme;

LAFA = (WW./(PPs.*Lt))-1;
LAFA12 = (WW - PPs.*Lt - PPps.*XP1 - XP2)./(PPs.*(Lt+XP1));

[WW(99,:)' LAFA(99,:)' XP1(99,:)' XP2(99,:)' LAFA12(99,:)']
[WW(154,:)' LAFA(154,:)' XP1(154,:)' XP2(154,:)' LAFA12(154,:)']