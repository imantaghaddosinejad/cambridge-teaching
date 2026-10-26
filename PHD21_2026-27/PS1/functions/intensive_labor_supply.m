function [nstar] = intensive_labor_supply(pbeta, pfrisch, peta, ptau, a, z, w, r, T, tol, maxiter)
% this solver maintains assumption that pfrisch<=1
% F(n) = n^(1/chi)*(n+m) - (1+beta)/eta has a unique root n* > max(0,-m) for any m

% newton finder parameters
if nargin < 10
    tol = 1e-8;
    maxiter = 100;
elseif nargin < 11
    maxiter = 100;
end

% safety checks (monotonicity)
m = (a*(1 + (1-ptau)*r) + T)/(z*w*(1-ptau));
f0 = -(1+pbeta)/peta;
f1 = 1 + m - (1+pbeta)/peta;
rootcond = f0*f1;

if f0 < 0 && rootcond > 0
    % solution exists outside bounds
    nstar = 1;
    return;
elseif f0 > 0
    error('No root exists for the given parameters. Please check the inputs.');
end

if pfrisch == 1
    % special case when pfrisch = 1 (quadratic rule)
    x = (-peta*m + sqrt((peta*m)^2 + 4*peta*(1+pbeta))) / (2*peta);
else
    % newton with bisection fallback
    % root bracketed: F < 0 on [0,-m], F(1) >= 0 (else returned above)
    lo = max(0, -m);
    hi = 1.00;
    x = hi;      
    err = 10;
    iter = 1;
    while err > tol && iter <= maxiter
        f = x^(1+1/pfrisch) + x^(1/pfrisch)*m-(1+pbeta)/peta;
        fp = (1+1/pfrisch)*x^(1/pfrisch) + x^(1/pfrisch-1)*m/pfrisch;

        % shrink bracket with sign of F
        if f > 0
            hi = x;
        else
            lo = x;
        end

        % newton step, bisect if it leaves the bracket (nests xp <= 0)
        xp = x - f/fp;
        if ~isfinite(xp) || xp <= lo || xp >= hi
            xp = 0.5*(lo + hi);
        end

        % update 
        err = abs(xp-x);
        x = xp;
        iter = iter + 1;
    end
    f = x^(1+1/pfrisch) + x^(1/pfrisch)*m-(1+pbeta)/peta;
    if ~(err <= tol) || ~(abs(f) <= 1e3*tol)
        % also catches err = NaN and false convergence off the root
        warning('newton not converged: err = %.3e | F = %.2e | iter = %d | z = %.4f', ...
            err, f, iter-1, z);
    end
end
nstar = x;