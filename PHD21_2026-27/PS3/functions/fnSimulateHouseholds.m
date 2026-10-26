%% Simulate Households - Aiyagari (1994) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Stochastic simulation of a panel of households. Initial states (a,z)
%   drawn from the stationary distribution, z pushed forward by Markov
%   draws, a pushed forward by the (interpolated) savings policy
%
%   Args:
%       mpolaprime: savings policy a'(a,z)
%       mcurrdist: stationary distribution over (a,z)
%       vgrida: asset grid
%       vgridz: productivity grid (levels)
%       mtransz: transition matrix of z
%       w: wage
%       r: interest rate
%       pnumhh: number of households N
%       pnumT: number of periods T (panel has T+1 columns)
%       seed: random seed
%
%   Returns:
%       ma: assets, N x (T+1), ma(:,1) drawn from mcurrdist
%       mz: productivity levels, N x (T+1)
%       mc: consumption, N x (T+1)
%       miz: productivity indices, N x (T+1)
%
function [ma, mz, mc, miz] = fnSimulateHouseholds(mpolaprime, mcurrdist, vgrida, vgridz, mtransz, w, r, pnumhh, pnumT, seed)

%==============================
% setup
%==============================
pnumgrida = numel(vgrida);
pnumgridz = numel(vgridz);
pnumstate = pnumgrida*pnumgridz;
vgrida = vgrida(:);
vgridz = vgridz(:);

ma = zeros(pnumhh,pnumT+1);
miz = zeros(pnumhh,pnumT+1); % z index
mc = zeros(pnumhh,pnumT+1);

%===============
% random draws
%===============
% column i belongs to household i: row 1 initial state, rows 2:T+1 markov draws
% path of household i fixed given seed (independent of pnumhh)
rng(seed);
mu = rand(pnumT+1,pnumhh)';

%===============
% initial states
%===============
% inverse cdf draw on flattened (a,z) distribution
vcdf = cumsum(mcurrdist(:));
vcdf = vcdf/vcdf(end); % normalise (rounding)

% F^{-1} is a step function (non-smooth), so we use the ">" operator to 
% find the first index where the cdf exceeds the uniform draw.
vidx = 1 + sum(vcdf' < mu(:,1), 2);
vidx(vidx>pnumstate) = pnumstate; % guard against rounding
[via, viz] = ind2sub([pnumgrida,pnumgridz], vidx);
ma(:,1) = vgrida(via);
miz(:,1) = viz;

%===============
% productivity paths
%===============
% exogenous, simulated once ahead of assets
mcumtransz = cumsum(mtransz,2);
mcumtransz(:,end) = 1; % guard rounding
for t = 1:pnumT
    miz(:,t+1) = 1 + sum(mcumtransz(miz(:,t),:) < mu(:,t+1), 2);
end
mz = vgridz(miz);

%===============
% asset paths
%===============
% a' interpolated in a, exact in z (off-grid assets)
for t = 1:pnumT+1
    [vlo,vhi,vwtlo,vwthi] = fnInterp1dVec(ma(:,t),vgrida);
    vidxlo = sub2ind(size(mpolaprime), vlo, miz(:,t));
    vidxhi = sub2ind(size(mpolaprime), vhi, miz(:,t));
    vaprime = vwtlo.*mpolaprime(vidxlo) + vwthi.*mpolaprime(vidxhi);
    vaprime(vaprime<vgrida(1)) = vgrida(1); % cap to stay on grid
    vaprime(vaprime>vgrida(end)) = vgrida(end); % cap to stay on grid 
    mc(:,t) = w*mz(:,t) + (1+r)*ma(:,t) - vaprime; % consumption

    % push forward (last a' only used for c at T+1)
    if t <= pnumT
        ma(:,t+1) = vaprime;
    end
end
end