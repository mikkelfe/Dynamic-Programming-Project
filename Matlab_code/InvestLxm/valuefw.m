function v = valuefw(s,c,t,x)
%VALFUNCW: value function
%s is set of all states
%x is (nn by 2), c is (nn by 1), s is (nn by 4), t is scalar

global basis n sminv smaxv smin smax e w T rp rn tcs tcb fme fds fdb si gamma0 EgRM theta

nn = size(s,1);
v = zeros(nn,1);
vval = zeros(nn,1);
TWval = zeros(nn,1);
K = size(e,1);

if (nargin==3)
   K=1; wk=1;
elseif ((s(:,3)+x(:,1))<1)       % this is for g(:,3)==0 or <1
   K=1;
end

for k = 1:K
   if (nargin==3)
      g = s;
   else
      if ((s(:,3)+x(:,1))<1)
         wk = 1;
      else
         wk = w(k);
      end
      ek = e(k,:);
      g = feval('gstate',s,x,ek);
   end
   gRM = exp(gamma0 + ek(:,3));
   		% g gives R, P, within bounds 
   		% Lt in bounds, but g3:L(t+1) is 0 or in bounds
   		% W can go out of bounds % W is computed from bounded R, P.                                    
   if g(:,3)<1                            % if g(:,3)==0,sell all land
      if theta == 0
         vval = (EgRM.^(T-t+1)).*s(:,4);
      elseif theta ==1
         vval = log(s(:,4)) + (gamma0.*(T-t+1));
      end
   else						%(2)
   	if isempty(c)		%(2.1)
      	TW = g(:,4);							% Terminal period net wealth
         vval = utility(TW);			      % Utility function
   	else					%(2.2)
      	gf4 = g(:,4);
      	nbri = find(gf4>0);              % No bankruptcy index
      	if isempty(nbri)==0   %(2.2.1)
      		gm4 = min(gf4,smaxv(4));
      		gd = gf4-gm4;   					% Out of bounds 4th state
      		g(:,4) = gm4;						% g(:,4) within upper bound
      		D = length(n);                 
      		phi = cell(1,D);
            for d=1:D
               if basis=='splibas'
                  phi{d} = feval(basis,n(d),smin(d),smax(d),g(nbri,d),0,1);
   				else
                  phi{d} = feval(basis,n(d),smin(d),smax(d),g(nbri,d));
               end
      		end
      		vval(nbri) = cdprodx(phi,c);  			% s(t+1)=g that computes v(s(t+1))
            TWval(nbri) = invutility(vval(nbri));
            
				ewi = find(gf4>smaxv(4));	% Positive land and more than Nmax state index
      		if isempty(ewi)==0   %(2.2.1..)
         		vval(ewi) = utility((TWval(ewi) + ...
                  ((gRM.^(T-t)).*gd(ewi))));
      		end
      	end

      	bri = find(gf4<=0);  		%Positive land and Bankruptcy index
         if isempty(bri)==0		%(2.2.2)
         	vval(bri) = utility(((1+rn).^(T-t)).*gf4(bri));
      	end
		end 			% end of: if isempty(c)
	end 				% end of: if g(:,3)==0
   v = v + vval.*wk;
end 					% end of: for k = 1:K