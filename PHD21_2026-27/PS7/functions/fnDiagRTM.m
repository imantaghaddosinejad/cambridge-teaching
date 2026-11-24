function res = fnDiagRTM(par, srceold, bc)
%% Diagnostics for the Krusell and Smith (1998) RCE solved by the repeated transition method
% inputs:
%   par     : parameters (palpha, pbeta, pdelta, peta, pmu, pnumgrida, pnumgridz,
%             pmina, pnumgridA, pathlength, burnin) and grids (vgrida, vgridz, mtransz)
%   srceold : srce for A=1.0 (steady-state K for scaling)
%   bc      : output of fnSolveBC_efficient / fnSolveBC_parallel (paths, policies, distribution)
% output:
%   res : law of motion coefficients and fit, den Haan (2010) errors, dynamic consistency,
%         monotonicity and Euler equation error statistics
%
% sections:
%   1. log-linear law of motion for K and w (state contingent on A)
%   2. den Haan (2010) dynamic forecast accuracy
%   3. dynamic consistency report
%   4. monotonicity check
%   5. Euler equation errors (one extra backward step at the converged solution)
%
% notes:
%   sample is the rtm window t = burnin+1,...,T-burnin (bc.twin)
%   section 5 uses the averaged (r',w',a'') expectation of fnSolveBC_efficient/_parallel
%
%% Setup
%==============================
% unpack
%==============================
% parameters
palpha      = par.palpha;
pbeta       = par.pbeta;
pdelta      = par.pdelta;
peta        = par.peta;
pmu         = par.pmu;
pnumgrida   = par.pnumgrida;
pnumgridz   = par.pnumgridz;
pmina       = par.pmina;
pnumgridA   = par.pnumgridA;
pathlength  = par.pathlength;
burnin      = par.burnin;

% grids and exogenous process
vgrida      = par.vgrida;
vgridz      = par.vgridz;
mtransz     = par.mtransz;
vgridA      = bc.vgridA;
mtransA     = bc.mtransA;
Kss         = srceold.K;

% equilibrium paths
tiA         = bc.tiA;
tK          = bc.tK;
tsupplyL    = bc.tsupplyL;
tr          = bc.tr;
tw          = bc.tw;
twin        = bc.twin;

%==============================
% sample (rtm window)
%==============================
pnumwin = numel(twin);
viA     = tiA(twin);       % A_t index
vK      = tK(twin);        % K_t
vKprime = tK(twin+1);      % K_{t+1}
vw      = tw(twin);        % w_t

%% 1. Log-linear law of motion
%===============
% capital: log K_{t+1} = beta0(A) + beta1(A) log K_t
%===============
mcoefK = NaN(pnumgridA,2);
vR2K   = NaN(pnumgridA,1);
vMSEK  = NaN(pnumgridA,1);
vnobs  = zeros(pnumgridA,1);
for iA = 1:pnumgridA
    vidx = (viA == iA);
    vnobs(iA) = sum(vidx);
    if vnobs(iA) >= 3
        X = [ones(vnobs(iA),1), log(vK(vidx))];
        y = log(vKprime(vidx));
        vb = X\y;
        vresid = y - X*vb;
        mcoefK(iA,:) = vb';
        vR2K(iA) = 1 - sum(vresid.^2)/sum((y-mean(y)).^2);
        vMSEK(iA) = mean(vresid.^2);
    end
end

% pooled fallback for rarely visited states
Xall = [ones(pnumwin,1), log(vK)];
vbpool = Xall\log(vKprime);
for iA = 1:pnumgridA
    if isnan(mcoefK(iA,1))
        mcoefK(iA,:) = vbpool';
    end
end

% one-step-ahead forecast (all states)
vlogKprime_hat = mcoefK(viA,1) + mcoefK(viA,2).*log(vK);
vresidK = log(vKprime) - vlogKprime_hat;
R2K_all = 1 - sum(vresidK.^2)/sum((log(vKprime)-mean(log(vKprime))).^2);
MSEK_all = mean(vresidK.^2);

%===============
% wage: log w_t = gamma0(A) + gamma1(A) log K_t
%===============
mcoefw = NaN(pnumgridA,2);
vR2w   = NaN(pnumgridA,1);
vMSEw  = NaN(pnumgridA,1);
for iA = 1:pnumgridA
    vidx = (viA == iA);
    if vnobs(iA) >= 3
        X = [ones(vnobs(iA),1), log(vK(vidx))];
        y = log(vw(vidx));
        vb = X\y;
        vresid = y - X*vb;
        mcoefw(iA,:) = vb';
        vR2w(iA) = 1 - sum(vresid.^2)/sum((y-mean(y)).^2);
        vMSEw(iA) = mean(vresid.^2);
    end
end
vlogw_hat = mcoefw(viA,1) + mcoefw(viA,2).*log(vK);
vresidw = log(vw) - vlogw_hat;
vresidw(isnan(vresidw)) = [];

% static (one-step) forecast errors in pct, by state
vKprime_static = exp(vlogKprime_hat);
verrK_static = 100*abs(vKprime_static - vKprime)./vKprime;
vmeanerr_static = accumarray(viA, verrK_static, [pnumgridA,1], @mean, NaN);

%===============
% print
%===============
fprintf('\n');
fprintf('======================== \n');
fprintf('Log-linear law of motion: log K'' = b0(A) + b1(A) log K \n');
fprintf('======================== \n');
fprintf('%4s %8s %6s %10s %10s %10s %12s %14s \n', ...
    'iA', 'A', 'nobs', 'b0', 'b1', 'R2', 'MSE', 'mean|err| (%)');
for iA = 1:pnumgridA
    fprintf('%4d %8.4f %6d %10.6f %10.6f %10.6f %12.3e %14.5f \n', ...
        iA, vgridA(iA), vnobs(iA), mcoefK(iA,1), mcoefK(iA,2), vR2K(iA), vMSEK(iA), vmeanerr_static(iA));
end
fprintf('all states: R2 = %.8f | MSE = %.3e | RMSE = %.5f%% \n', R2K_all, MSEK_all, 100*sqrt(MSEK_all));

fprintf('\n');
fprintf('======================== \n');
fprintf('Log-linear wage rule: log w = g0(A) + g1(A) log K \n');
fprintf('======================== \n');
fprintf('%4s %8s %10s %10s %10s %12s \n', 'iA', 'A', 'g0', 'g1', 'R2', 'MSE');
for iA = 1:pnumgridA
    fprintf('%4d %8.4f %10.6f %10.6f %10.6f %12.3e \n', ...
        iA, vgridA(iA), mcoefw(iA,1), mcoefw(iA,2), vR2w(iA), vMSEw(iA));
end
fprintf('all states: MSE = %.3e \n', mean(vresidw.^2));

%% 2. den Haan (2010) dynamic forecast
%===============
% iterate the rule on its own predictions: Khat_1 = K_1
%===============
vKdyn = zeros(pnumwin,1);
vKdyn(1) = vK(1);
for it = 1:pnumwin-1
    vKdyn(it+1) = exp(mcoefK(viA(it),1) + mcoefK(viA(it),2)*log(vKdyn(it)));
end
verrK_dyn = 100*abs(vKdyn - vK)./vK;
[maxerr_dyn, imax_dyn] = max(verrK_dyn);
[maxerr_static, imax_static] = max(verrK_static);

%===============
% print
%===============
fprintf('\n');
fprintf('======================== \n');
fprintf('den Haan (2010) accuracy: 100*|Khat-K|/K \n');
fprintf('======================== \n');
fprintf('dynamic forecast: max = %.5f%% | mean = %.5f%% \n', maxerr_dyn, mean(verrK_dyn));
fprintf('  max at t = %d (A = %.4f, K = %.4f) \n', twin(imax_dyn), vgridA(viA(imax_dyn)), vK(imax_dyn));
fprintf('static forecast:  max = %.5f%% | mean = %.5f%% \n', maxerr_static, mean(verrK_static));
fprintf('  max at t = %d (A_t = %.4f, A_t+1 = %.4f, K = %.4f) \n', twin(imax_static)+1, ...
    vgridA(viA(imax_static)), vgridA(tiA(twin(imax_static)+1)), vKprime(imax_static));

%===============
% plot
%===============
vsample = (1:min(500,pnumwin))'; % first 500 periods of window
figure;
subplot(2,1,1);
plot(twin(vsample), vK(vsample), 'r-', 'LineWidth', 1.5);
hold on;
plot(twin(vsample)+1, vKprime_static(vsample), 'b--', 'LineWidth', 1.2);
plot(twin(vsample), vKdyn(vsample), 'k-.', 'LineWidth', 1.2);
hold off;
grid on;
ylabel('K');
legend('Equilibrium', 'One-step forecast', 'Dynamic forecast (den Haan)', ...
    'Location', 'best', 'FontSize', 9);
subplot(2,1,2);
plot(twin, verrK_dyn, 'k-', 'LineWidth', 1);
hold on;
plot(twin+1, verrK_static, 'b-', 'LineWidth', 1);
hold off;
grid on;
xlim([twin(1), twin(end)+1]);
xlabel('Period');
ylabel('100|Khat-K|/K');
legend('Dynamic', 'One-step', 'Location', 'best', 'FontSize', 9);
set(gcf, 'Units', 'inches', 'Position', [1 1 8 6]);
exportgraphics(gcf, '../figures/rtm_lom_K.png', 'Resolution', 300);

figure;
plot(twin(vsample), vw(vsample), 'r-', 'LineWidth', 1.5);
hold on;
plot(twin(vsample), exp(vlogw_hat(vsample)), 'b--', 'LineWidth', 1.2);
hold off;
grid on;
xlabel('Period');
ylabel('w');
legend('Equilibrium', 'Log-linear rule', 'Location', 'best', 'FontSize', 9);
set(gcf, 'Units', 'inches', 'Position', [1 1 8 4]);
exportgraphics(gcf, '../figures/rtm_lom_w.png', 'Resolution', 300);

%% 3. Dynamic consistency report
%===============
% realised (forward simulation) vs predicted (guess) paths
%===============
verrcK = bc.tK_new(twin) - tK(twin);
verrcK_pct = 100*verrcK/Kss;

fprintf('\n');
fprintf('======================== \n');
fprintf('Dynamic consistency report for rtm \n');
fprintf('======================== \n');
fprintf('max absolute error (pct of steady-state K): %.6f%% \n', max(abs(verrcK_pct)));
fprintf('root mean squared error (pct of steady-state K): %.6f%% \n', sqrt(mean(verrcK_pct.^2)));

figure;
histogram(verrcK_pct, 100);
grid on;
xlabel('Dynamic consistency error (pct of steady-state K)');
ylabel('Frequency');
set(gcf, 'Units', 'inches', 'Position', [1 1 5 4]);
exportgraphics(gcf, '../figures/rtm_consistency.png', 'Resolution', 300);

%% 4. Monotonicity check
%===============
% consumption at a fixed individual state against K, by A
%===============
% result is robust to the choice of individual state
ia = floor((pnumgrida+1)/2); % median asset grid point
iz = floor((pnumgridz+1)/2); % median productivity
vspearman = NaN(pnumgridA,1);

figure;
for iA = 1:pnumgridA
    vidx = (viA == iA);
    vKs = vK(vidx);
    vcs = squeeze(bc.tmpolc(ia,iz,twin(vidx)));
    if numel(vKs) >= 3
        vspearman(iA) = corr(vKs, vcs, 'Type', 'Spearman');
    end
    subplot(2,ceil(pnumgridA/2),iA);
    scatter(vKs, vcs, 10, 'filled');
    grid on;
    xlabel('K');
    ylabel('c(a,z;S)');
    title(sprintf('A = %.4f', vgridA(iA)));
end
set(gcf, 'Units', 'inches', 'Position', [1 1 12 6]);
exportgraphics(gcf, '../figures/rtm_monotonicity.png', 'Resolution', 300);

fprintf('\n');
fprintf('======================== \n');
fprintf('Monotonicity check: c(a=%.3f, z=%.3f; K, A) \n', vgrida(ia), vgridz(iz));
fprintf('======================== \n');
for iA = 1:pnumgridA
    fprintf('A = %.4f: Spearman rank corr(K, c) = %.6f \n', vgridA(iA), vspearman(iA));
end

%% 5. Euler equation errors
%===============
% one extra backward step at the converged solution: EE = c_new/c - 1
%===============
mgridz = repmat(vgridz',pnumgrida,1);
mgrida = repmat(vgrida',1,pnumgridz);
mgridiz = repmat(1:pnumgridz,pnumgrida,1); % z index
mtranszrow = mtransz(mgridiz(:),:);         % (a,z) x z' transition probabilities
titransp = [(2:pathlength)'; pathlength];
viAcan = bc.viAcan;

% counterfactual candidates for each future tfp state, sorted by K
ccanloc = cell(pnumgridA,1);
ccanK = cell(pnumgridA,1);
for iAp = 1:pnumgridA
    vcanloc = find(tiA == viAcan(iAp));
    vcanloc = vcanloc(vcanloc >= burnin+1 & vcanloc <= pathlength-burnin);
    [vcanK, vidx] = sort(tK(vcanloc));
    ccanloc{iAp} = vcanloc(vidx);
    ccanK{iAp} = vcanK;
end

% backward step over the window
tEE = zeros(pnumgrida,pnumgridz,pnumwin);
for it = 1:pnumwin
itrans = twin(it);

% current aggregate state and prices
iA = tiA(itrans);
w = tw(itrans);
r = tr(itrans);

% future aggregate state (for beliefs)
ifuture = titransp(itrans);
iAfuture = tiA(ifuture);
Kprime = tK(itrans+1);

% converged policy and its location on the wealth grid
mpolaprime = bc.tmpolaprime(:,:,itrans);
vaprime = mpolaprime(:);
[mloq, mhiq, mwloq, mwhiq] = fnInterp1dVec(mpolaprime, vgrida, true); % linear + extrapolation

% expectations (rhs euler)
mexp = 0;
for iAp = 1:pnumgridA
    Aprime = vgridA(iAp);

    % periods that represent the future state (A',K')
    if iAp == iAfuture
        tlow = ifuture;
        thigh = ifuture;
        wtlow = 1;
    else
        vcanK = ccanK{iAp};
        vcanloc = ccanloc{iAp};
        ilow = sum(vcanK < Kprime);
        ilow(ilow<=1) = 1;
        ilow(ilow>=numel(vcanK)) = numel(vcanK)-1;
        ihigh = ilow+1;
        wtlow = (vcanK(ihigh) - Kprime)/(vcanK(ihigh) - vcanK(ilow));
        wtlow(wtlow<0) = 0;
        wtlow(wtlow>1) = 1;
        tlow = vcanloc(ilow);
        thigh = vcanloc(ihigh);
    end

    % future prices: average over candidates
    K2Lprime_low = Kprime/tsupplyL(tlow);
    K2Lprime_high = Kprime/tsupplyL(thigh);
    rprime = wtlow*(palpha*Aprime*K2Lprime_low^(palpha-1) - pdelta) ...
        + (1-wtlow)*(palpha*Aprime*K2Lprime_high^(palpha-1) - pdelta);
    wprime = wtlow*((1-palpha)*Aprime*K2Lprime_low^palpha) ...
        + (1-wtlow)*((1-palpha)*Aprime*K2Lprime_high^palpha);

    % future savings: average over candidates
    mpolaprime_low = bc.tmpolaprime(:,:,tlow);
    mpolaprime_high = bc.tmpolaprime(:,:,thigh);
    aprimeprime_low = mwloq(:).*mpolaprime_low(mloq(:),:) + mwhiq(:).*mpolaprime_low(mhiq(:),:);
    aprimeprime_high = mwloq(:).*mpolaprime_high(mloq(:),:) + mwhiq(:).*mpolaprime_high(mhiq(:),:);
    aprimeprime = wtlow*aprimeprime_low + (1-wtlow)*aprimeprime_high;

    % marginal adjustment cost psi_2(a'',a') and future adjustment cost
    psi2 = -(pmu/2)*((aprimeprime./vaprime).^2 - 1);
    adjprime = (pmu/2)*((aprimeprime-vaprime)./vaprime).^2.*vaprime;

    % future labour and consumption
    mprime = ((1+rprime)*vaprime - adjprime - aprimeprime)./(wprime*vgridz');
    nprime = (-peta*mprime + sqrt((peta*mprime).^2 + 4*peta))/(2*peta);
    nprime(nprime>1) = 1;
    cprime = wprime*vgridz'.*nprime + (1+rprime)*vaprime - aprimeprime - adjprime;
    cprime(cprime<1e-10) = 1e-10;
    muprime = 1./cprime;

    % update expectations
    mexpfut = sum(mtranszrow.*(1+rprime-psi2).*muprime, 2);
    mexp = mexp + mtransA(iA,iAp)*reshape(mexpfut, pnumgrida, pnumgridz);
end
mexp = pbeta*mexp;
mexp = mexp./(1 + pmu*(mpolaprime-mgrida)./mgrida); % marginal adjustment cost today psi_1(a',a)

% implied consumption
c = 1./(mexp + bc.tmlambda(:,:,itrans));
c(c<1e-10) = 1e-10;

% euler error (relative consumption change)
tEE(:,:,it) = c./bc.tmpolc(:,:,itrans) - 1;
end

%===============
% summary statistics
%===============
tlogEE = log10(max(abs(tEE),1e-16));
twdist = bc.tcurrdist(:,:,twin);
logEE_mean = mean(tlogEE,'all');
logEE_max = max(tlogEE,[],'all');
logEE_wmean = log10(sum(twdist.*abs(tEE),'all')/pnumwin); % distribution-weighted mean |EE|

% fixed individual state and state-average series
vlogEE_state = squeeze(tlogEE(ia,iz,:));
vlogEE_avg = log10(squeeze(mean(abs(tEE),[1 2])));

fprintf('\n');
fprintf('======================== \n');
fprintf('Euler equation errors (log10 |c_new/c - 1|) \n');
fprintf('======================== \n');
fprintf('mean over (a,z,t): %.4f | max: %.4f \n', logEE_mean, logEE_max);
fprintf('distribution-weighted mean |EE|: %.4f \n', logEE_wmean);
fprintf('state (a=%.3f, z=%.3f): mean %.4f | max %.4f \n', ...
    vgrida(ia), vgridz(iz), mean(vlogEE_state), max(vlogEE_state));

%===============
% plot
%===============
figure;
subplot(2,3,1);
scatter(twin, vlogEE_state, 8);
grid on;
xlabel('Period');
ylabel('EE (log_{10}), fixed (a,z)');
subplot(2,3,2);
histogram(vlogEE_state, 100);
grid on;
xlabel('EE (log_{10}), fixed (a,z)');
subplot(2,3,3);
scatter(vK, vlogEE_state, 8);
grid on;
xlabel('K');
ylabel('EE (log_{10}), fixed (a,z)');
subplot(2,3,4);
scatter(twin, vlogEE_avg, 8);
grid on;
xlabel('Period');
ylabel('mean EE (log_{10})');
subplot(2,3,5);
histogram(vlogEE_avg, 100);
grid on;
xlabel('mean EE (log_{10})');
subplot(2,3,6);
scatter(vK, vlogEE_avg, 8);
grid on;
xlabel('K');
ylabel('mean EE (log_{10})');
set(gcf, 'Units', 'inches', 'Position', [1 1 12 6]);
exportgraphics(gcf, '../figures/rtm_ee.png', 'Resolution', 300);

%% Results
%==============================
% law of motion
%==============================
res.mcoefK          = mcoefK;
res.vR2K            = vR2K;
res.vMSEK           = vMSEK;
res.vnobs           = vnobs;
res.R2K_all         = R2K_all;
res.MSEK_all        = MSEK_all;
res.mcoefw          = mcoefw;
res.vR2w            = vR2w;
res.vMSEw           = vMSEw;

%==============================
% den Haan (2010)
%==============================
res.vKdyn           = vKdyn;
res.vKprime_static  = vKprime_static;
res.maxerr_dyn      = maxerr_dyn;
res.meanerr_dyn     = mean(verrK_dyn);
res.maxerr_static   = maxerr_static;
res.meanerr_static  = mean(verrK_static);

%==============================
% dynamic consistency, monotonicity, euler errors
%==============================
res.maxerr_consist  = max(abs(verrcK_pct));
res.rmse_consist    = sqrt(mean(verrcK_pct.^2));
res.vspearman       = vspearman;
res.logEE_mean      = logEE_mean;
res.logEE_max       = logEE_max;
res.logEE_wmean     = logEE_wmean;

end
