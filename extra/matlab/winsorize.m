function a_out = winsorize (a_in, pct)

if (nargin == 1) 
    pct = [1,99];
elseif (length(pct) == 1)
    pct = [pct, 100 - pct];
end

p = prctile (a_in(:), pct);
a_out = a_in;
a_out(a_out < p(1)) = p(1);
a_out(a_out > p(2)) = p(2);
