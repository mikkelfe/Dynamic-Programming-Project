% Error terms: generating randomly
clear all;

Ee = zeros(1,1);							% Mean of error terms for R and P eqns	
VarCov = zeros(1,1);				      % Variance-covariance matrix
VarCov(1,1) = 0.026941;
TRIALS = 500;
YEARS = 20;
TY = TRIALS*YEARS;
Em3 = MVNRND(Ee,VarCov,TY);
Er3 = reshape(Em3,TRIALS,YEARS);
%E3 = cell(1,YEARS);

load errors;

for YR = 1:YEARS
   E123{YR} = [E{YR} Er3(:,YR)];
end

save errors3 E123;