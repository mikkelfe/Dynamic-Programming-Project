function u = utility(y)
% Utility function

global b theta

if theta==0
   u = y;
elseif theta==1
   u = (log(b)./b).*y;
	byi = find(y>=b);
   u(byi) = log(y(byi));
else
   disp('utility function for theta=0 or 1');
end