function y = invutility(u)
% Inverse of: utility function

global b theta

if theta==0
   y = u;
elseif theta==1
   y = (u.*b)./log(b);
   bui = find(u>=log(b));
   y(bui) = exp(u(bui));
else
   disp('utility function for theta=0 or 1');
end