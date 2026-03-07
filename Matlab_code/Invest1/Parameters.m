% Investment Decision Model:
%Obj. func: 
%          Max: EU(W(T+1))
clear all;
tic;
%profile on -detail builtin
%___________________________________________________________________________

global basis beta0 beta1 alpha0 alpha1 alpha2 cost sminv smaxv smin smax...
   n rp rn tcs tcb fme fds q qf e w T theta b valuef bound dau

% Parameters

file1 = 'su77521.txt';			   % Saving in text file:
file2 = 'su77521';				   % Saving the work space

n1=7; n2=7; n3=5; n4=21;		   % Number of nodes for each state
                                 % Number of x levels used to max Bellman's eqn
%q = 81; qf = 0;                  % one-stage hybrid method
q = 41; qf = 21;				% two-stage	hybrid method
                                 
%basis = 'chebbas'; node = 'chebnode';		% Chebychev polynomial basis functions and nodes
basis = 'splibas'; node = 'nodeunif';     % Linear Spline basis functions and nodes 
%___________________________
if qf==0
   vmaxh = 'vmaxh1';
else
   vmaxh = 'vmaxh2';
end

valuef = 'valuefw';			      % Value function for max: wealth(T+1)
bound = 'boundi';					   % Incomplete market with borrowing constraint
%___________________________
m1=5; m2=5;						      % Number of nodes for approximating integration-
										   % for E[v(error)]: use m>=(n+1)/2 for accuracy
theta = 0;							   % Utility function parameter
b = 60000;
T = 19;          					   % Time horizon
t1 = 1;
beta0 = 1.51181;   					% Parameters for R state eqn
beta1 = 0.742391;
alpha0 = 0.215; 					% Parameters for P state eqn
alpha1 = 0.908361;	 				% alpha in stateq.wf1: made AR1
alpha2 = 0.079432;
Ee = zeros(2,1);						% Mean of error terms for R and P eqns	
VarCov = zeros(2,2);				   % Variance-covariance matrix
VarCov(1,1) = 0.030186;
VarCov(2,2) = 0.017292;
cost = 231.0;                    % cost of production+management per acre
Rmin = 230.0; Rmax = 540.0; 	   % Gross Return per acre on composite crop	
Pmin = 1010.0; Pmax = 2840.0; 	% Price of land (dollars)				
Lmin = 400.0; Lmax = 2000.0;	   % Land (acres)
Wmin = 0; Wmax = 6000000;
sminv = [Rmin Pmin Lmin Wmin];   % All states in a vector
smaxv = [Rmax Pmax Lmax Wmax];
n = [n1 n2 n3 n4];
m = [m1 m2];
if node=='chebnode'
   smin = sminv - ((sminv-smaxv)./(2.*(cos((n-1+0.5).*pi./n)))+(sminv-smaxv)./2);
   smax = smaxv + ((sminv-smaxv)./(2.*(cos((n-1+0.5).*pi./n)))+(sminv-smaxv)./2);
else
   smin = sminv;
   smax = smaxv;
end

rp = 0.03;     					       % Lending:Interest rate on positive amount	
rn = rp + 0.03; 					   % Borrowing:Interest rate on negative amount
tcs = 0.06;  						   % Transaction cost (%) on selling  farmland
tcb = 0.01;                            % Transaction cost (%) on purchasing farmland
fme = 300;							   % Farm machinery equipment(fme) $ per acre
fds = 0.07;					 		   % fme selling deduction/transcation cost (%)							% fme buying deduction (%)
dau = 0.7;							   % Debt-to-asset ratio constraint (upper bound)