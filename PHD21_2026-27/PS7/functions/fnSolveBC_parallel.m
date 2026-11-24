function res = fnSolveBC_parallel(par, srceold, opt)
%% Solve Krusell and Smith (1998) RCE with endogenous labour and wealth adjustment costs
% repeated transition method (Lee, 2025), precomputed counterfactual search
% inputs:
%   par     : parameters (palpha, pbeta, pdelta, pfrisch, peta, pmu, pnumgrida, pnumgridz,
%             pmina, prhoA, psigA, pnumgridA, pathlength, burnin) and grids from fnSolveSS
%             (vgrida, vgridz, mtransz)
%   srceold : srce for A=1.0 (start distribution, K, initial policies)
%   opt     : macros (verbose1, verbose2, iterdisp, vecdist)
% output:
%   res : simulated equilibrium paths (shocks, aggregates, prices) and convergence
%
% notes:
%   log utility (sigma = 1) and adjustment cost psi(a',a) = (pmu/2)*((a'-a)/a)^2*a
%   psi_1(a',a) = d psi/d a' = pmu*(a'-a)/a
%   psi_2(a',a) = d psi/d a  = -(pmu/2)*((a'/a)^2 - 1)
%   next-period labour solved in closed form, valid for pfrisch = 1 only
%   t=1 anchored to srceold (distribution, K); K(T+1) snapped to srceold.K
%   A is held at its period-T value from T+1 onwards
%   errors and counterfactual candidates restricted to t = burnin+1,...,T-burnin
%   differences from fnSolveBC:
%   (1) counterfactual search done once per ge iteration, before the backward solution
%   (2) future prices (r',w') and savings a'' averaged over the two candidates,
%       expectations computed once per A'
%   (3) backward solution in parfor: period t only uses last iteration objects
%       (tmpolaprime, tmlambda, tK, tsupplyL) and the lookup, so periods are independent
%
%% Model Fundamentals
%==============================
% unpack
%==============================
% parameters
palpha      = par.palpha;
pbeta       = par.pbeta;
pdelta      = par.pdelta;
pfrisch     = par.pfrisch;
peta        = par.peta;
pmu         = par.pmu;
pnumgrida   = par.pnumgrida;
pnumgridz   = par.pnumgridz;
pmina       = par.pmina;
prhoA       = par.prhoA;
psigA       = par.psigA;
pnumgridA   = par.pnumgridA;
pathlength  = par.pathlength;
burnin      = par.burnin;

% grids and exogenous process (from fnSolveSS)
vgrida      = par.vgrida;
vgridz      = par.vgridz;
mtransz     = par.mtransz;

% boundary conditions
startdist   = srceold.mcurrdist;
Kss         = srceold.K;
Lss         = srceold.supplyL;

% macros
verbose1  = opt.verbose1;
verbose2  = opt.verbose2;
iterdisp  = opt.iterdisp;
vecdist   = opt.vecdist;

%==============================
% aggregate productivity (TFP)
%==============================
pmuA = 0.0;
[mtransA, vgridA] = fnTauchen(prhoA, pmuA, psigA^2, pnumgridA, 3.0); % psigA is innovation sd
vgridA = exp(vgridA);

%==============================
% simulate aggregate shock path
%==============================
pseed = 100;
rng(pseed);
iAinit = ceil(pnumgridA/2);                   % A = 1.0 at t=1 (consistent with srceold)
tiA = fnSimShock(mtransA, pathlength, iAinit); % shock index path
tA = vgridA(tiA);                              % tfp level path
titransp = [(2:pathlength)'; pathlength];     % future time index (A fixed at A_T from T+1)
twin = ((burnin+1):(pathlength-burnin))';     % error and candidate window

% counterfactual candidate state (fallback if a tfp state is rarely visited in window)
vnumvisit = accumarray(tiA(twin), 1, [pnumgridA,1]);
viAcan = (1:pnumgridA)';
for iAp = 1:pnumgridA
    if vnumvisit(iAp) < 2
        vvalid = find(vnumvisit >= 2);
        [~,inear] = min(abs(vvalid - iAp));
        viAcan(iAp) = vvalid(inear);
        warning('tfp state %d visited %d times in window: candidates taken from state %d', ...
            iAp, vnumvisit(iAp), viAcan(iAp));
    end
end

%% Numerical Setup
%==============================
% equilibrium objects
%==============================
tmpolc          = repmat(srceold.mpolc,1,1,pathlength);
tmpoln          = repmat(srceold.mpoln,1,1,pathlength);
tmpolaprime     = repmat(srceold.mpolaprime,1,1,pathlength);
tmpolaprime_new = repmat(srceold.mpolaprime,1,1,pathlength);
tmlambda        = repmat(srceold.mlambda,1,1,pathlength);
tmlambda_new    = repmat(srceold.mlambda,1,1,pathlength);

% counterfactual lookup: (t, A') -> candidate periods below/above K' and weight on below
tilow  = zeros(pathlength,pnumgridA);
tihigh = zeros(pathlength,pnumgridA);
twtlow = zeros(pathlength,pnumgridA);

%==============================
% auxiliary objects
%==============================
mgridz = repmat(vgridz',pnumgrida,1);
mgrida = repmat(vgrida',1,pnumgridz);
mgridiz = repmat(1:pnumgridz,pnumgrida,1); % z index
mzoffset = (mgridiz-1)*pnumgrida;          % linear index offset for z columns
mtranszrow = mtransz(mgridiz(:),:);         % (a,z) x z' transition probabilities

%==============================
% numerical parameters
%==============================
tolbc = 1e-6;
maxiter = 3000;
wt1 = 0.9000;
wt2 = 0.9000;
wt3 = 0.9000;
wt4 = 0.9000;

%==============================
% initial guess
%==============================
tK = Kss*ones(pathlength+1,1) + 1e-4*randn(pathlength+1,1); % noise avoids ties in candidate search
tK(1) = Kss;                % predetermined
tK(pathlength+1) = Kss;     % snap end to srce
tsupplyL = Lss*ones(pathlength,1);

%==============================
% parallel pool
%==============================
% thread-based pool: shared memory, no copying of policy paths to workers
if isempty(gcp('nocreate'))
    parpool('Threads');
end

%% Numerical Solution
%===============
% outer ge loop
%===============
hfig = figure('Visible', verbose2); % hidden unless plotting interim progress

iter = 1;
err = 10;
timerbc = tic;
while err > tolbc && iter <= maxiter

%===============
% update price path
%===============
tK2L = tK(1:pathlength)./tsupplyL;
tr = palpha.*tA.*tK2L.^(palpha-1) - pdelta;
tw = (1-palpha).*tA.*tK2L.^palpha;

%%
%===============
% counterfactual search (given last iteration tK)
%===============
% step 1: candidate periods for each future tfp state, sorted by K
ccanloc = cell(pnumgridA,1);
ccanK = cell(pnumgridA,1);
for iAp = 1:pnumgridA
    vcanloc = find(tiA == viAcan(iAp));
    vcanloc = vcanloc(vcanloc >= burnin+1 & vcanloc <= pathlength-burnin);
    [vcanK, vidx] = sort(tK(vcanloc));
    ccanloc{iAp} = vcanloc(vidx);
    ccanK{iAp} = vcanK;
end

% step 2: for each (t, A'), candidates closest to K' from below and above
for itrans = 1:pathlength
    Kprime = tK(itrans+1);
    for iAp = 1:pnumgridA
        vcanK = ccanK{iAp};
        vcanloc = ccanloc{iAp};
        ilow = sum(vcanK < Kprime);
        ilow(ilow<=1) = 1;
        ilow(ilow>=numel(vcanK)) = numel(vcanK)-1;
        ihigh = ilow+1;
        wtlow = (vcanK(ihigh) - Kprime)/(vcanK(ihigh) - vcanK(ilow));
        wtlow(wtlow<0) = 0;
        wtlow(wtlow>1) = 1;

        % store
        tilow(itrans,iAp) = vcanloc(ilow);
        tihigh(itrans,iAp) = vcanloc(ihigh);
        twtlow(itrans,iAp) = wtlow;
    end
end

%%
%===============
% backward solution
%===============
% parallel over periods (order irrelevant: no dependence on this iteration's policies)
parfor itrans = 1:pathlength

% current aggregate state and prices
iA = tiA(itrans);
w = tw(itrans);
r = tr(itrans);

% future aggregate state (for beliefs)
ifuture = titransp(itrans);
iAfuture = tiA(ifuture);
Kprime = tK(itrans+1);

% counterfactual lookup for this period (row slices)
vilow = tilow(itrans,:);
vihigh = tihigh(itrans,:);
vwtlow = twtlow(itrans,:);

% current guess and its location on the wealth grid (reused for all future states)
mpolaprime = tmpolaprime(:,:,itrans);
vaprime = mpolaprime(:);
[mloq, mhiq, mwloq, mwhiq] = fnInterp1dVec(mpolaprime, vgrida, true); % linear + extrapolation

% expectations (rhs euler, rational expectations over A' and z')
mexp = 0;
for iAp = 1:pnumgridA
    Aprime = vgridA(iAp);

    % periods that represent the future state (A',K')
    if iAp == iAfuture && itrans < pathlength
        % realised future state: on the path
        tlow = ifuture;
        thigh = ifuture;
        wtlow = 1;
    else
        % counterfactual future state: precomputed lookup
        tlow = vilow(iAp);
        thigh = vihigh(iAp);
        wtlow = vwtlow(iAp);
    end

    % future prices: average over candidates (K' realised, L' from candidate periods)
    K2Lprime_low = Kprime/tsupplyL(tlow);
    K2Lprime_high = Kprime/tsupplyL(thigh);
    rprime = wtlow*(palpha*Aprime*K2Lprime_low^(palpha-1) - pdelta) ...
        + (1-wtlow)*(palpha*Aprime*K2Lprime_high^(palpha-1) - pdelta);
    wprime = wtlow*((1-palpha)*Aprime*K2Lprime_low^palpha) ...
        + (1-wtlow)*((1-palpha)*Aprime*K2Lprime_high^palpha);

    % future savings for all z' at once: (a,z) x z', average over candidates
    mpolaprime_low = tmpolaprime(:,:,tlow);
    mpolaprime_high = tmpolaprime(:,:,thigh);
    aprimeprime_low = mwloq(:).*mpolaprime_low(mloq(:),:) + mwhiq(:).*mpolaprime_low(mhiq(:),:);
    aprimeprime_high = mwloq(:).*mpolaprime_high(mloq(:),:) + mwhiq(:).*mpolaprime_high(mhiq(:),:);
    aprimeprime = wtlow*aprimeprime_low + (1-wtlow)*aprimeprime_high;

    % marginal adjustment cost psi_2(a'',a') and future adjustment cost
    psi2 = -(pmu/2)*((aprimeprime./vaprime).^2 - 1);
    adjprime = (pmu/2)*((aprimeprime-vaprime)./vaprime).^2.*vaprime;

    % future labour (closed form intratemporal foc, pfrisch = 1)
    mprime = ((1+rprime)*vaprime - adjprime - aprimeprime)./(wprime*vgridz');
    nprime = (-peta*mprime + sqrt((peta*mprime).^2 + 4*peta))/(2*peta);
    nprime(nprime>1) = 1;

    % future consumption
    cprime = wprime*vgridz'.*nprime + (1+rprime)*vaprime - aprimeprime - adjprime;
    cprime(cprime<1e-10) = 1e-10;
    muprime = 1./cprime;

    % update expectations (sum over z', weight on A')
    mexpfut = sum(mtranszrow.*(1+rprime-psi2).*muprime, 2);
    mexp = mexp + mtransA(iA,iAp)*reshape(mexpfut, pnumgrida, pnumgridz);
end
mexp = pbeta*mexp;
mexp = mexp./(1 + pmu*(mpolaprime-mgrida)./mgrida); % marginal adjustment cost today psi_1(a',a)

% optimal intratemporal policies
c = 1./(mexp + tmlambda(:,:,itrans));
c(c<1e-10) = 1e-10;
n = (w*mgridz./(peta*c)).^pfrisch;
n(n<0) = 0;
n(n>1) = 1;
madj = (pmu/2)*((mpolaprime-mgrida)./mgrida).^2.*mgrida;
mpolaprime_newtemp = w*mgridz.*n + (1+r)*mgrida - c - madj;
mlambda_newtemp = 1./(w*mgridz.*n + (1+r)*mgrida - mpolaprime - madj) - mexp;
mlambda_newtemp(mlambda_newtemp<0) = 0; % economically relevant sign

% constraints (kkt)
mlambda_newtemp(mpolaprime_newtemp>pmina) = 0; % constraint slack
mpolaprime_newtemp(mpolaprime_newtemp<=pmina) = pmina;

% update
tmpolc(:,:,itrans) = c; % jump-update
tmpoln(:,:,itrans) = n;
tmpolaprime_new(:,:,itrans) = mpolaprime_newtemp;
tmlambda_new(:,:,itrans) = mlambda_newtemp;
end

%%
%===============
% forward simulation
%===============
% non-stochastic histogram method
tcurrdist = zeros(pnumgrida,pnumgridz,pathlength);
tcurrdist(:,:,1) = startdist; % anchored to srceold

% vectorised (two-step push forward)
if vecdist
    % interpolate entire policy path on grid once per outer iteration
    [tlo, thi, twlo, twhi] = fnInterp1dVec(tmpolaprime_new, vgrida);

    for itrans = 1:pathlength-1
        % current distribution
        currdist = tcurrdist(:,:,itrans);

        % step 1: (a,z) -> (a',z)
        mlo = tlo(:,:,itrans) + mzoffset;
        mhi = thi(:,:,itrans) + mzoffset;
        mwlo = twlo(:,:,itrans);
        mwhi = twhi(:,:,itrans);
        vidx = [mlo(:); mhi(:)];
        vval = [currdist(:).*mwlo(:); currdist(:).*mwhi(:)];
        mmid = reshape(accumarray(vidx, vval, [pnumgrida*pnumgridz,1]), pnumgrida, pnumgridz);

        % step 2: (a',z) -> (a',z')
        tcurrdist(:,:,itrans+1) = mmid*mtransz;
    end
% non-vectorised
else
    for itrans = 1:pathlength-1
        % next distribution (point-wise)
        currdist = tcurrdist(:,:,itrans);
        mnextdist = zeros(pnumgrida,pnumgridz);
        for iz = 1:pnumgridz
        for ia = 1:pnumgrida
            nexta = tmpolaprime_new(ia,iz,itrans);
            [lo,hi,wlo,whi] = fnInterp1dGrid(nexta,vgrida,pnumgrida); % interpolate aprime on grid
            mass = currdist(ia,iz);

            for izp = 1:pnumgridz
                mnextdist(lo,izp) = mnextdist(lo,izp) + mass*mtransz(iz,izp)*wlo;
                mnextdist(hi,izp) = mnextdist(hi,izp) + mass*mtransz(iz,izp)*whi;
            end
        end
        end
        tcurrdist(:,:,itrans+1) = mnextdist;
    end
end

%===============
% aggregates (mcc)
%===============
tK_new = [squeeze(sum(tcurrdist.*mgrida,[1 2])); Kss];             % capital supply (K(T+1) snapped)
tsupplyL_new = squeeze(sum(tcurrdist.*tmpoln.*mgridz,[1 2]));      % effective labour
thours = squeeze(sum(tcurrdist.*tmpoln,[1 2]));                    % average hours
tlambda = squeeze(sum(tcurrdist.*tmlambda,[1 2]));
tlambda_new = squeeze(sum(tcurrdist.*tmlambda_new,[1 2]));

%===============
% update (ge)
%===============
% error (window excludes burnin at both ends)
err = mean(abs([ ...
    tK_new(twin) - tK(twin); ...
    tsupplyL_new(twin) - tsupplyL(twin); ...
    tlambda_new(twin) - tlambda(twin)].^2));

% update
tK          = wt1.*tK          + (1-wt1).*tK_new;
tsupplyL    = wt2.*tsupplyL    + (1-wt2).*tsupplyL_new;
tmpolaprime = wt3.*tmpolaprime + (1-wt3).*tmpolaprime_new;
tmlambda    = wt4.*tmlambda    + (1-wt4).*tmlambda_new;
tK(1)            = Kss; % predetermined
tK(pathlength+1) = Kss; % snap end to srce

%===============
% interim report
%===============
timelapse = toc(timerbc);
if verbose1 && mod(iter, iterdisp)==0 || iter == 1 || err <= tolbc

    % print
    fprintf(' \n');
    fprintf('** RTM Results ** \n');
    fprintf('iter %d (%.2f mins): error = %.15f \n', iter, timelapse/60, err);
    fprintf('max |dK|: %.8f   |  max |dL|: %.8f   (window) \n', ...
        max(abs(tK_new(twin) - tK(twin))), max(abs(tsupplyL_new(twin) - tsupplyL(twin))));

    % save (mid)
    dir = '../solutions/WIP_bc_parallel.mat';
    save(dir, 'tK', 'tsupplyL', 'tiA', 'iter', 'err');

    % plot
    if verbose2
        set(0,'CurrentFigure',hfig); % plot to own figure even if another is clicked
        subplot(1,2,1);
        plot(1:pathlength, tK(1:pathlength), 'b-', 'LineWidth', 1);
        hold on;
        plot(1:pathlength, tK_new(1:pathlength), 'r-.', 'LineWidth', 1);
        hold off;
        yline(Kss, 'k--');
        xline(burnin, 'k:');
        xline(pathlength-burnin, 'k:');
        xlim([1,pathlength]);
        xlabel('Period');
        ylabel('K');
        legend('Predicted', 'Realised', 'Location', 'northeast','FontSize',12);
        subplot(1,2,2);
        plot(1:pathlength, tsupplyL, 'b-', 'LineWidth', 1);
        hold on;
        plot(1:pathlength, tsupplyL_new, 'r-.', 'LineWidth', 1);
        hold off;
        yline(Lss, 'k--');
        xline(burnin, 'k:');
        xline(pathlength-burnin, 'k:');
        xlim([1,pathlength]);
        xlabel('Period');
        ylabel('L');
        drawnow;
    end
end
iter = iter + 1;
end % outer ge loop

%==============================
% equilibrium allocations
%==============================
tK2L = tK(1:pathlength)./tsupplyL;
tr = palpha.*tA.*tK2L.^(palpha-1) - pdelta;
tw = (1-palpha).*tA.*tK2L.^palpha;
tY = tA.*tK(1:pathlength).^palpha.*tsupplyL.^(1-palpha);
tC = squeeze(sum(tcurrdist.*tmpolc,[1 2]));
tI = tK(2:pathlength+1) - (1-pdelta)*tK(1:pathlength);
tmadj = (pmu/2)*((tmpolaprime_new-mgrida)./mgrida).^2.*mgrida;
tADJ = squeeze(sum(tcurrdist.*tmadj,[1 2]));

%==============================
% print result
%==============================
fprintf('\n**** RTM Results ****\n');
fprintf('iter %d (%.2f mins): error = %.15f \n', iter-1, timelapse/60, err);
fprintf('mean K = %.4f | mean L = %.4f | mean hours = %.4f (window) \n', ...
    mean(tK(twin)), mean(tsupplyL(twin)), mean(thours(twin)));
fprintf('resource check (window): max |Y-C-I-ADJ| = %.2e \n', ...
    max(abs(tY(twin)-tC(twin)-tI(twin)-tADJ(twin))));

%==============================
% results
%==============================
% exogenous process and shock path
res.vgridA      = vgridA;
res.mtransA     = mtransA;
res.tiA         = tiA;
res.tA          = tA;
res.twin        = twin;

% aggregates and prices
res.tK          = tK;
res.tK_new      = tK_new;
res.tsupplyL    = tsupplyL;
res.thours      = thours;
res.tY          = tY;
res.tC          = tC;
res.tI          = tI;
res.tADJ        = tADJ;
res.tr          = tr;
res.tw          = tw;

% policies, multipliers and distribution (for diagnostics)
res.tmpolaprime = tmpolaprime;  % converged guess used for beliefs
res.tmpolc      = tmpolc;
res.tmpoln      = tmpoln;
res.tmlambda    = tmlambda;
res.tcurrdist   = tcurrdist;
res.viAcan      = viAcan;

% convergence
res.err         = err;
res.iter        = iter-1;
res.converged   = err <= tolbc;

%==============================
% save (final)
%==============================
dir = '../solutions/bc_parallel.mat';
save(dir, 'res', 'par');

end
