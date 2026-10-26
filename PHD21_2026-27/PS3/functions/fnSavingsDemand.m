%% Savings Demand - Aiyagari (1994) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Given r, solve household policies (pfi) to convergence, then the
%   stationary distribution (histogram method) under converged policies
%
%   Args:
%       r: interest rate (wage from firm foc)
%       par: struct of parameters, grids and numerical parameters
%
%   Returns:
%       Ea: aggregate savings
%       flag: [1, 1] if policy and distribution converged, [0, 0] otherwise
%
function [Ea, flag] = fnSavingsDemand(r, par)

%==============================
% unpack parameters
%==============================
pbeta = par.pbeta;
palpha = par.palpha;
pdelta = par.pdelta;
pA = par.pA;
pnumgrida = par.pnumgrida;
pnumgridz = par.pnumgridz;
pmina = par.pmina;
mtransz = par.mtransz;
vgridz = par.vgridz;
vgrida = par.vgrida;
tolpol = par.tolpol;
toldist = par.toldist;
maxiter = par.maxiter;
wt2 = par.wt2;
wt3 = par.wt3;

%==============================
% equilibrium objects
%==============================
mpolaprime = repmat(vgrida'.*0.2,1,pnumgridz);
mpolaprime_new = zeros(pnumgrida,pnumgridz);
mlambda = zeros(pnumgrida,pnumgridz);
mlambda_new = zeros(pnumgrida,pnumgridz);
mcurrdist = ones(pnumgrida,pnumgridz) / (pnumgrida*pnumgridz);

%==============================
% auxiliary objects
%==============================
mgridz = repmat(vgridz',pnumgrida,1);
mgrida = repmat(vgrida',1,pnumgridz);
mgridiz = repmat(1:pnumgridz,pnumgrida,1); % z index

%===============
% prices
%===============
% given r, firm foc pins down K2L and w (unless w fixed)
if isfield(par, 'w')
    w = par.w;
else
    K2L = ((r+pdelta)/(palpha*pA))^(1/(palpha-1)); % requires r > -pdelta
    w = (1-palpha)*pA*K2L^palpha;
end

% future state/prices
rprime = r;
wprime = w;

%===============
% policy function iteration (to convergence)
%===============
errpol = 10.0;
iterpol = 1;
while errpol > tolpol && iterpol <= maxiter

    % expectations (rhs euler, rational expectations)
    mexp = 0;
    for izp = 1:pnumgridz
        % future consumption
        aprimeprime = interp1(vgrida', mpolaprime(:,izp), mpolaprime, "linear", "extrap");
        cprime = wprime*vgridz(izp) + (1+rprime)*mpolaprime - aprimeprime;
        cprime(cprime<1e-10) = 1e-10;
        muprime = 1./cprime;

        % update expectations
        mexp = mexp + mtransz(:,izp)'.*(1+rprime).*muprime;
    end
    mexp = pbeta*mexp;

    % optimal intratemporal policies
    c = 1./(mexp + mlambda);
    c(c<1e-10) = 1e-10;
    mpolaprime_new = w*mgridz + (1+r)*mgrida - c;
    mlambda_new = 1./(w*mgridz + (1+r)*mgrida - mpolaprime) - mexp;
    mlambda_new(mlambda_new<0) = 0; % economically relevant sign

    % constraints (nnc, kkt)
    mlambda_new(mpolaprime_new>pmina+1e-10) = 0; % constraint slack (tol avoids knife-edge)
    mpolaprime_new(mpolaprime_new<pmina) = pmina;

    % error
    errpol = max(abs([mpolaprime_new-mpolaprime; mlambda_new-mlambda]),[],"all");

    % update
    mpolaprime = wt2.*mpolaprime + (1-wt2).*mpolaprime_new;
    mlambda    = wt3.*mlambda    + (1-wt3).*mlambda_new;
    iterpol = iterpol + 1;
end

%===============
% stationary distribution 
%===============
% histogram evolving method - non-stochastic
% vectorised (sparse transition matrix)
pnumstate = pnumgrida*pnumgridz;
[mlo,mhi,mwtlo,mwthi] = fnInterp1dVec(mpolaprime_new,vgrida); % aprime on grid (fixed policy)

% step 1: (a,z) -> (aprime,z), block diagonal in z
vfrom = (1:pnumstate)';
vlo = mlo(:) + (mgridiz(:)-1)*pnumgrida; % linear index of (lo,z)
vhi = mhi(:) + (mgridiz(:)-1)*pnumgrida; % linear index of (hi,z)
mTa = sparse([vfrom;vfrom], [vlo;vhi], [mwtlo(:);mwthi(:)], pnumstate, pnumstate);

% step 2: (aprime,z) -> (aprime,zprime)
mTz = kron(mtransz, speye(pnumgrida));

% full transition (rows sum to 1)
mT = mTa*mTz;

% iterate
errdist = 10;
iterdist = 1;
vcurrdist = mcurrdist(:);
while errdist > toldist && iterdist <= maxiter
    vnextdist = mT'*vcurrdist;
    errdist = max(abs(vnextdist-vcurrdist));
    vcurrdist = vnextdist;
    iterdist = iterdist + 1;
end
mcurrdist = reshape(vcurrdist,pnumgrida,pnumgridz);

%===============
% savings demand
%===============
Ea = sum(mcurrdist.*mpolaprime_new,"all");
flag = [double(errpol <= tolpol), double(errdist <= toldist)];
end