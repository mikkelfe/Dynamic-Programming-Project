function g = gstate(s,x,ek);

% State transition function: states s, control x, error term ek

global beta0 beta1 alpha0 alpha1 alpha2 cost rn rp tcs tcb fme fds sminv smaxv gamma0 EgRM

[nn d]= size(s);
g = zeros(size(s));
gf1 = exp(beta0 + (beta1.*log(s(:,1))) + ek(:,1));  							% R state eqn
gf2 = exp(alpha0 + (alpha1.*log(s(:,2))) + alpha2.*log(s(:,1)) + ek(:,2));	% P state eqn
g(:,1) = max(sminv(1),min(gf1,smaxv(1)));
g(:,2) = max(sminv(2),min(gf2,smaxv(2)));
g(:,3) = s(:,3) + x(:,1);										% Land state eqn
gRM = exp(gamma0 + ek(:,3));

s2bs = ((1+tcb).*s(:,2)) + fme;
nxi = find(x(:,1)<0);
s2bs(nxi) = ((1-tcs).*s(nxi,2)) + (1-fds).*fme;
s2v = (1-tcs).*s(:,2) + (1-fds).*fme;		% For normalization
g2v = (1-tcs).*g(:,2) + (1-fds).*fme;		%  "        "
At = s(:,4) - (s2v.*s(:,3));
r = ones(size(s,1),1).*rp;

if g(:,3)<1                         % this is for if g(:,3)==0, sell all land
   g(:,4) = EgRM.*s(:,4);
else
   ia = At - (s2bs.*x(:,1)) - x(:,2) - (cost.*g(:,3)) ;
   nai = find(ia<0);  					% Negative amount index
   r(nai) = rn;
   gAt = ((1+r).*ia) + (g(:,1).*g(:,3)) + (gRM.*x(:,2));
   g(:,4) = gAt + g2v.*g(:,3);
end