function [xml,xmu] = boundm(s,xqi)

% Bound function

global sminv smaxv dau fme tcs tcb fds


if ((s(:,3)+xqi(:,1))<1)
   xml = s(:,4);
   xmu = s(:,4);
else
   s2bs = ((1+tcb).*s(:,2)) + fme;
   nxi = find(xqi(:,1)<0);
   s2bs(nxi) = ((1-tcs).*s(nxi,2)) + (1-fds).*fme;
   s2v = (1-tcs).*s(:,2) + (1-fds).*fme;		% For normalization
   %g2v = (1-tcs).*g(:,2) + (1-fds).*fme;		%  "        "
   At = s(:,4) - (s2v.*s(:,3));
   xml = zeros(size(s,1),1);
   xmu = max(0,(At - (s2bs.*xqi) + dau.*s2v.*(s(:,3) + xqi)));
end