function [xl,xu] = boundi(s);
% Bound function for one control

global sminv smaxv dau fme tcs tcb fds

xl = sminv(3)-s(:,3);			% Bounds from Land state			
xu3 = smaxv(3)-s(:,3);
s2v = (1-tcs).*s(:,2) + (1-fds).*fme;	% selling price: for normalization 
s2b = ((1+tcb).*s(:,2)) + fme;         % buying price

xdm = max(0,((s(:,4) - (1-dau).*s2v.*s(:,3))./(s2b - dau.*s2v)));
xu = min(xu3,xdm);
