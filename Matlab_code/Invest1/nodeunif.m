function x = nodeunif(n,a,b)
if numel(n)==1
    if n==1
        x = (a+b)/2;
    else
        x = linspace(a,b,n)';
    end
else
    error('This fallback nodeunif only supports scalar n.');
end
