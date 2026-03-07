% Investment Decision Model:
%Obj. func: 
%          Max: EU(W(T+1))
clear all;
tic;
%profile on -detail builtin
%___________________________________________________________________________

global basis beta0 beta1 alpha0 alpha1 alpha2 cost sminv smaxv smin smax...
   n rp rn tcs tcb fme fds q q2 e w T theta b valuef bound dau gamma0 EgRM

% Parameters

file1 = 'su77521Lxmth1.txt';			   % Saving in text file:
file2 = 'su77521Lxmth1';				   % Saving the work space

n1=7; n2=7; n3=5; n4=21;	   % Number of nodes for each state
                                 % Number of x levels used to max Bellman's eqn

q = 41; q2 = 25;					   % 
                                 
%basis = 'chebbas'; node = 'chebnode';		% Chebychev polynomial basis functions and nodes
basis = 'splibas'; node = 'nodeunif';     % Linear Spline basis functions and nodes 
%
%___________________________
m1=3; m2=3;	m3=3;				      % Number of nodes for approximating integration-
										   % for E[v(error)]: use m>=(n+1)/2 for accuracy
theta = 1;							   % Utility function parameter
b = 60000;
T = 19;          					   % Time horizon
t1 = 1;
beta0 = 1.51181;   					% Parameters for R state eqn
beta1 = 0.742391;
alpha0 = 0.215; 					% Parameters for P state eqn
alpha1 = 0.908361;	 				% alpha in stateq.wf1: made AR1
alpha2 = 0.079432;
gamma0 = 0.057757;
EgRM = 1.073826;
Ee = zeros(3,1);						% Mean of error terms for R and P eqns	
VarCov = zeros(3,3);				   % Variance-covariance matrix
VarCov(1,1) = 0.030186;
VarCov(2,2) = 0.017292;
VarCov(3,3) = 0.026941;
cost = 231.0;                    % cost of production+management per acre
Rmin = 230.0; Rmax = 540.0; 	   % Gross Return per acre on composite crop	
Pmin = 1010.0; Pmax = 2840.0; 	% Price of land (dollars)				
Lmin = 400.0; Lmax = 2000.0;	   % Land (acres)
Wmin = 0; Wmax = 6000000;
sminv = [Rmin Pmin Lmin Wmin];   % All states in a vector
smaxv = [Rmax Pmax Lmax Wmax];
n = [n1 n2 n3 n4];
m = [m1 m2 m3];
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