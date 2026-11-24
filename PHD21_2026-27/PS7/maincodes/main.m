%% Cambridge PHD21 Computational Methods 2026-27
% Problem Set 7
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
opt.verbose2 = false;   % plot interim progress (off for srce, on for tce)
opt.verbose3 = false;   % warm start
opt.iterdisp = 200;     % number of iterations per progress report
opt.vecdist = true;     % vectorised histogram method

%% Part a. Solve KS (1998) SRCE for A=1.0, calibrate peta
%==============================
% parameters
%==============================
par.palpha = 0.33;
par.pbeta = 0.99;
par.pdelta = 0.025;
par.pfrisch = 1.00;     % chi
par.pmu = 0.60;         % adjustment cost
par.prhoz = 0.90;
par.psigz = 0.05;       % innovation sd log z
par.pnumgrida = 100;
par.pnumgridz = 7;
par.pmina = 1.0;        % borrowing limit
par.pmaxa = 200;
par.pcurve = 7;
par.peta = 7.836111;    % calibrated manually: average hours = 1/3 at A=1.0

% stationary competitive equilibrium for A=1.0
par.pA = 1.0;
srceold = fnSolveSS(par, opt);

%% Part b. Solve KS (1998) SRCE for A=1.1, peta fixed
% stationary competitive equilibrium for A=1.1
par.pA = 1.1;
srcenew = fnSolveSS(par, opt);

% grids and exogenous process (identical across srce)
par.vgrida = srceold.vgrida;
par.vgridz = srceold.vgridz;
par.mtransz = srceold.mtransz;

%% Part d. KS (1998) TCE for A=1.0 -> A=1.1
par.pA = 1.1; % new (permanent) tfp
par.pathlength = 2000;
tA = ones(par.pathlength,1)*par.pA;

opt.iterdisp = 50;      % number of iterations per progress report
opt.verbose2 = true;    % plot tce interim progress
tce = fnSolveTCE(par, srceold, srcenew, tA, opt);

% plot
figure;
subplot(1,2,1);
plot(1:par.pathlength, tce.tK, 'b-', 'LineWidth', 1.6);
yline(srcenew.K, 'k--', 'Label', 'New SRCE');
yline(srceold.K, 'k:', 'Label', 'Old SRCE');
xlabel('Period');
ylabel('K');
subplot(1,2,2);
plot(1:par.pathlength, tce.tsupplyL, 'b-', 'LineWidth', 1.6);
yline(srcenew.supplyL, 'k--', 'Label', 'New SRCE');
xlabel('Period');
ylabel('L');
exportgraphics(gcf, '../figures/fig_d_tce.png', 'Resolution', 300);

%% Part g. KS (1998) RCE via repeated transition method
par.prhoA = 0.95;
par.psigA = 0.009;      % innovation sd log A
par.pnumgridA = 7;
par.pathlength = 3001;  % total simulated length incl. burnin (10000 for final run)
par.burnin = 500;       % dropped at both ends of the path

opt.iterdisp = 25;      % number of iterations per progress report (and plot)
opt.verbose2 = true;    % plot rtm interim progress
opt.vecdist = true;     % vectorised histogram method
% bc = fnSolveBC(par, srceold, opt);           % sequential search (no policies returned)
% bc = fnSolveBC_efficient(par, srceold, opt); % precomputed search, averaged r', w', a''
bc = fnSolveBC_parallel(par, srceold, opt);    % as efficient, backward solution in parfor

%% Parts h & i. Law of motion, den Haan (2010), dynamic consistency, monotonicity, Euler errors
rtmdiag = fnDiagRTM(par, srceold, bc);

%% Parts j & k. Business cycle statistics
%==============================
% series (rtm window, logs)
%==============================
twin = bc.twin;
vnames = {'Y'; 'I'; 'C'; 'L'};
mlogx = log([bc.tY(twin), bc.tI(twin), bc.tC(twin), bc.tsupplyL(twin)]);

% hp filter (cyclical component)
[~, mloghp] = hpfilter(mlogx, Smoothing=1600);

%==============================
% moments: volatility, autocorrelation, cyclicality
%==============================
cseries = {mlogx, mloghp};
ctitles = {'Part j. Business cycle statistics: log series', ...
    'Part k. Business cycle statistics: log and HP-filtered (lambda = 1600)'};
cstats = cell(2,1);
for is = 1:2
    mx = cseries{is};
    vsd = 100*std(mx)';                 % volatility (pct)
    vsdrel = vsd/vsd(1);                % relative to output
    vac = zeros(numel(vnames),1);
    vcorrY = zeros(numel(vnames),1);
    for iv = 1:numel(vnames)
        vac(iv) = corr(mx(1:end-1,iv), mx(2:end,iv)); % first-order autocorrelation
        vcorrY(iv) = corr(mx(:,iv), mx(:,1));         % co-movement with output
    end
    cstats{is} = table(vsd, vsdrel, vac, vcorrY, 'RowNames', vnames, ...
        'VariableNames', {'SD (pct)', 'SD / SD(Y)', 'Autocorr', 'Corr with Y'});

    % print
    fprintf('\n');
    fprintf('======================== \n');
    fprintf('%s \n', ctitles{is});
    fprintf('======================== \n');
    disp(cstats{is});
end
bcstats_log = cstats{1};
bcstats_hp = cstats{2};
