%Approximation:
%     Hybrid method: Discretize x space to maximize Bellman's eqn
%		Using basis function and nodes for value
% 		function v(s)=c*phi(s),and for policy function use c(t+1) and solve				
%METHOD: 
%		(To find value function)
%		1.Specify state nodes in T (which are aslo for all t)
%     2.Maximize Bellman's eqn for each node to find optimal x(T) from contol space
%		3.Get	x(T)=argmax Bellman's, v(T)=max Bellman's for each node s
%		4.For T, solve c=v/phi since # of equations = # of unknowns 	
%		5.Update c by making iterations of redoing steps 2-4 for all t
%
%_____________________________________________________________________________

Parameters;
fid = fopen(file1,'w+t'); 		% Creating a file for output
										% w+t for deleting old contents, and for 
                              %reading and writing new output in text
fprintf(fid,'Max: E[U(Net Wealth(T+1))]\n');
fprintf(fid,'Number of nodes for states = %3i %3i %3i %3i\n',n);
fprintf(fid,'Number of x levels to find max = %3i %3i\n',q,qf);
fprintf(fid,'Theta in utility func = %g\n',theta);
fprintf(fid,'Time horizon T = %3i\n',T);
fprintf(fid,'Number of nodes for error terms = %3i %3i\n\n',m);

										%g prints the number in compact notation
                           	%3i is for printing integer which holds 3 digits
                           	% \n is for starting new line in printing
yearnorm = ' Year  norm1ratio  norm2ratio  max&mean(abs(x-xold)) p(=<1)';
fprintf(yearnorm)					% For displaying
fprintf(fid,yearnorm);			% For saving in the file
%___________________________________________________________________________
[e w]=qnwnorm(m,Ee,VarCov);		% Normal distribution of error terms
D = length(n);						   % Computes number of states
phiinv = cell(1,D);              % basis matrices at nodes for each state
si = cell(1,D);                  % nodes
for d=1:D
   si{d} =  feval(node,n(d),smin(d),smax(d));	% (n(d) by 1)
   if basis=='splibas'
      phiinv{d} = inv(feval(basis,n(d),smin(d),smax(d),si{d},0,1)); %Linear spline
   else
      phiinv{d} = inv(feval(basis,n(d),smin(d),smax(d),si{d}));
   end
end
st = cgrid(si);
pn = size(st,1);                 % pn=prod(n)=n1*n2*n3*n4
copt = zeros(pn,T);					% Matrix for c for 1:T
                         % First node w=0, others are big amounts>1
wi = find(st(:,4)>1);   % To avoid rounding error for w>0
s = st(wi,:);
nn = size(s,1);
      
% Algorithm for 1:T (for finding value function)
c = [];
x = -s(:,3);						   % x in T+1
for t=T:-1:t1
   xold = x;                  	% Store old value for comparing 
   [x,v] = feval(vmaxh,s,c,t);  	% Solve Bellman equation at nodes.
   vt = ones(pn,1).*utility(0);  % v=u(0) for w=0
   vt(wi) = v;                    % opt solution for w>0 
   c=ckronx(phiinv,vt);				% Coefficients c for value function in each t
   copt(:,t) = c;						% Store them in the matrix (nn x T)
   change = (x-xold);
   mchange = max(abs(change));	% Compute maximum change
   achange = mean(abs(change));
   n1change = (norm(change,1))/(norm(xold,1));
   n2change = (norm(change))/(norm(xold));   
   pchange = 100.*size(find(abs(change)<=1),1)/size(change,1);
	fprintf('\n%3i\t %4.6f\t %4.6f\t %6.4f\t %6.4f\t %6.4f\n',...
      t,n1change,n2change,mchange,achange,pchange)	% For displaying
   fprintf(fid,'\n%3i\t %4.6f\t %4.6f\t %6.4f\t %6.4f\t %6.4f\n',...
      t,n1change,n2change,mchange,achange,pchange);
end
clear phiinv s st change mchange achange pchange wi v vt c x xold d D yearnorm
%________________________________________________________________________________
toc;     						% Elapsed time since tic was used
seconds = toc;
fprintf(fid,'\nCPU Time seconds= %15.2f\n',toc); % For saving in the file
fclose(fid);    					% Returns 0 if successful in closing output file
save(file2);
disp('For output, run result**.m');
%profile report invest;
