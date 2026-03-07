% Error terms: generating randomly
clear all;

Ee = zeros(2,1);							% Mean of error terms for R and P eqns	
VarCov = zeros(2,2);				      % Variance-covariance matrix
VarCov(1,1) = 0.030186;
VarCov(2,2) = 0.017292;
TRIALS = 500;
YEARS = 20;
TY = TRIALS*YEARS;
Em = MVNRND(Ee,VarCov,TY);
Em1 = Em(:,1);
Em2 = Em(:,2);
Er1 = reshape(Em1,TRIALS,YEARS);
Er2 = reshape(Em2,TRIALS,YEARS);
E = cell(1,YEARS);
for YR = 1:YEARS
   E{YR} = [Er1(:,YR) Er2(:,YR)];
end

%save errors
save errors E;
