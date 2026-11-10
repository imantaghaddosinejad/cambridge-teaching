%% Cambridge PHD21 Computational Methods 2026-27
% Problem Set 4
% Lecturer: Dr. Hanbaek Lee (https://sites.google.com/view/hanbaeklee/)
% TA: Iman Taghaddosinejad (https://github.com/imantaghaddosinejad)
%%
%==============================
% housekeeping 
%==============================
clear;
close all;
clc;
addpath('../functions');
addpath('../figures');
addpath('../solutions');

%==============================
% macros
%==============================
opt.verbose1 = true;    % report interim progress
opt.verbose2 = true;    % plot interim progress
opt.verbose3 = false;   % warm start
opt.iterdisp = 100;      % number of iterations per progress report
opt.vecdist = true;     % vectorised histogram method

%% Parts a & b. Solve Aiyagari (1994) SRCE
%==============================
% parameters (Aiyagari, 1994)
%==============================
par.pbeta = 0.96;
par.palpha = 0.36;
par.pdelta = 0.08;
par.pnumgrida = 100;
par.pnumgridz = 7;
par.pmina = 0;
par.pmaxa = 300;
par.pcurve = 2;

% stationary competitive equilibrium for A=1.0
par.pA = 1.0;
srceold = fnSolveGE(par, opt);

% stationary competitive equilibrium for A=1.1
par.pA = 1.1;
srcenew = fnSolveGE(par, opt);

%% Part d. Aiyagari (1994) TCE
par.pA = 1.1; % new (permanent) tfp shock  
par.pathlength = 551;
tA = ones(par.pathlength,1)*par.pA;

opt.iterdisp = 50;      % number of iterations per progress report
tce = fnSolveTCE(par, srceold, srcenew, tA, opt);

%% Parts e & f. Transition with GE effects muted (prices fixed at new srce)
%==============================
% setup 
%==============================
pathlengthpe = par.pathlength;
pnumgrida = par.pnumgrida;
pnumgridz = par.pnumgridz;
vgrida = srcenew.vgrida;
mtransz = srcenew.mtransz;
mgridiz = repmat(1:pnumgridz,pnumgrida,1);  % z index
mzoffset = (mgridiz-1)*pnumgrida;           % linear index offset for z columns
tolconv = 1e-3;                             % convergence criterion (relative K gap)

% fixed policy: new srce policy every period (prices fixed at new srce)
mpolaprime = srcenew.mpolaprime;
[mlo, mhi, mwlo, mwhi] = fnInterp1dVec(mpolaprime, vgrida);
mlo = mlo + mzoffset;
mhi = mhi + mzoffset;
vidx = [mlo(:); mhi(:)];

%==============================
% forward simulation 
%==============================
tcurrdistpe = zeros(pnumgrida,pnumgridz,pathlengthpe);
tcurrdistpe(:,:,1) = srceold.mcurrdist;     % old srce distribution
tKpe = zeros(pathlengthpe,1);
tKpe(1) = srceold.K;                        % predetermined

for itrans = 1:pathlengthpe-1
    % current distribution 
    currdist = tcurrdistpe(:,:,itrans);

    % step 1: (a,z) -> (a',z)
    vval = [currdist(:).*mwlo(:); currdist(:).*mwhi(:)];
    mmid = reshape(accumarray(vidx, vval, [pnumgrida*pnumgridz,1]), pnumgrida, pnumgridz);

    % step 2: (a',z) -> (a',z')
    tcurrdistpe(:,:,itrans+1) = mmid*mtransz;

    % aggregates 
    tKpe(itrans+1) = sum(currdist.*mpolaprime,'all');
end

%==============================
% convergence dates: (e) GE vs (f) PE 
%==============================
tgapge = abs(tce.tK - srcenew.K)/srcenew.K;
tgappe = abs(tKpe - srcenew.K)/srcenew.K;
tconvge = find(tgapge > tolconv, 1, 'last') + 1;
tconvpe = find(tgappe > tolconv, 1, 'last') + 1;

% print 
fprintf('periods to converge (|K-K*|/K* < %.0e): GE = %d | PE = %d \n', tolconv, tconvge, tconvpe);

% plot 
figure;
plot(1:pathlengthpe, tce.tK, 'b-', 'LineWidth', 1.6);
hold on;
plot(1:pathlengthpe, tKpe, 'r-.', 'LineWidth', 1.6);
yline(srcenew.K, 'k--');
hold off;
xlabel('Period');
ylabel('K');
legend('GE (part e)', 'PE: prices fixed at new SRCE (part f)', ...
    'Location', 'southeast','FontSize',12);
exportgraphics(gcf, '../figures/fig_f_ge_vs_pe.png', 'Resolution', 300);

%% Part g. Gini along the transition path
%==============================
% gini per period (histogram) 
%==============================
pathlength = numel(tce.tK);
vgrida = srcenew.vgrida;
tgini = zeros(pathlength,1);

for itrans = 1:pathlength
    marga = sum(tce.tcurrdist(:,:,itrans),2);   % marginal wealth distribution
    vwealth = vgrida'.*marga;                   % wealth held at each grid point
    vpop = [0; cumsum(marga)];                  % cumulative population share
    vlor = [0; cumsum(vwealth)/sum(vwealth)];   % cumulative wealth share
    tgini(itrans) = 1 - 2*trapz(vpop, vlor);    % gini = 1 - 2*(area under lorenz)
end

% steady-state benchmarks 
ginissold = tgini(1);                           % Phi_1 = old srce
margnew = sum(srcenew.mcurrdist,2);
vwealthnew = vgrida'.*margnew;
ginissnew = 1 - 2*trapz([0; cumsum(margnew)], [0; cumsum(vwealthnew)/sum(vwealthnew)]);

% print 
fprintf('gini: old srce %.4f | t=2 %.4f | new srce %.4f | end of path %.4f \n', ...
    ginissold, tgini(2), ginissnew, tgini(end));

% plot 
figure;
plot(1:pathlength, tgini, 'b-', 'LineWidth', 1.6);
yline(ginissold, 'k:', 'Label', 'Old SRCE');
yline(ginissnew, 'k--', 'Label', 'New SRCE');
xlabel('Period');
ylabel('Wealth Gini');
exportgraphics(gcf, '../figures/fig_g_gini_path.png', 'Resolution', 300);

%% Part h. MIT shock (5% fall in TFP)
par.pA = 1.00; % transitory tfp shock  
par.pathlength = 551;
tA = ones(par.pathlength,1)*par.pA;
tA(1) = 0.95;

% old and new steady-states 
srceold = fnSolveGE(par, opt);
srcenew = srceold;

% run TCE 
opt.iterdisp = 50;      % number of iterations per progress report
tce = fnSolveTCE(par, srceold, srcenew, tA, opt);

%% Part i. MIT shock with decay (5% fall in TFP, with decay recovery)
prhoA = 0.90;
par.pA = 1.00; % long-run tfp
tlogA = zeros(par.pathlength,1);
tlogA(1) = log(0.95); % initial 5% fall at t=1
for itrans = 2:par.pathlength
    tlogA(itrans) = prhoA*tlogA(itrans-1);
end
tA = exp(tlogA); % tfp level path

% old and new steady-states 
srceold = fnSolveGE(par, opt);
srcenew = srceold;

% run TCE 
opt.iterdisp = 50;      % number of iterations per progress report
tce = fnSolveTCE(par, srceold, srcenew, tA, opt);