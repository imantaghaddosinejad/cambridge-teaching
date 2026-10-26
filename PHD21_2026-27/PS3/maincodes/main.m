%% Cambridge PHD21 Computational Methods 2026-27
% Problem Set 3
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
verbose1 = true;    % report interim progress
verbose2 = true;    % plot interim progress 
verbose3 = false;   % warm start
iterdisp = 25;      % number of iterations per progress report
vecdist = true;     % vectorised histogram method

%% Part e. Solve Aiyagari (1994) SRCE
%==============================
% parameters (Aiyagari, 1994) 
%==============================
pbeta = 0.96;
palpha = 0.36;
pdelta = 0.08;
pA = 1.0;
pnumgrida = 200;
pnumgridz = 7;
pmina = 0;
pmaxa = 300;
pcurve = 2;
solvege;

%% Part f. Replicate Figures IIa and IIb in Aiyagari (1994) 
CapDemandSupply;

%% Part g. Simulate Panel of Households
pnumhh = 10000; % number of households
pnumT = 100; % number of periods
seed = 12345; % random seed
[ma, mz, mc, miz] = ...
    fnSimulateHouseholds(mpolaprime_new, mcurrdist, vgrida, vgridz, mtransz, w, r, pnumhh, pnumT, seed);

% aggregate wealth series 
twealth = mean(ma,1);
sd_wealth = std(twealth); 
cv_wealth = sd_wealth/mean(twealth);
fprintf('(N,T) = (%d, %d) | mean wealth: %.4f | sd: %.4f | cv: %.4f | sd/K: %.4f \n', ...
    pnumhh, pnumT, mean(twealth), sd_wealth, cv_wealth, sd_wealth/K);
figure;
plot(0:pnumT,twealth,'LineWidth', 1.6);
yline(K,'k--','Label','Ea','FontSize', 10);
ylabel('Aggregate Wealth');
xlabel('Period');
exportgraphics(gcf, '../figures/fig_g_aggregate_wealth.png', 'Resolution', 300);

% Comments: 
% Fluctuations are finite-N sampling noise (no aggregate risk in the
% continuum economy) and shrinking as 1/√N.
% The noise is persistent, since wealth at the top is near unit root.
% A short window (T) understates the sampling sd. A long window recovers it.
% The small downward gap below the RCE K and wealth is numerical. The
% histogram method splits a' onto grid nodes, the simulation keeps exact
% (interpolated) a', so their stationary distributions differ slightly.
% The gap shrinks with a finer grid and builds slowly over T (persistence).

%% Part h. Bottom 10% of Households (by wealth) Dynamics
% wealth dynamics of the bottom 10% (wealth)  at t=50 
tcan = 50;
[~, vsort] = sort(ma(:,tcan+1));
vidxp10 = vsort(1:round(0.1*pnumhh));             % indices of bottom 10% at t = 50
vt = tcan+1:tcan+51;                              % columns t = 50,...,100
mcohort = ma(vidxp10, vt);                        % cohort panel, 1000 x 51
tmeanp10 = mean(mcohort,1);
mbandp10 = prctile(mcohort, [2.5 97.5], 1);       % 2 x 51: lower, upper
figure;
hold on;
fill([0:50, fliplr(0:50)], [mbandp10(1,:), fliplr(mbandp10(2,:))], ...
    [0.7 0.8 1.0], 'EdgeColor', 'none', 'FaceAlpha', 0.5); % 95% band
plot(0:50, tmeanp10, 'b-', 'LineWidth', 1.6);
plot(0:50, twealth(vt), 'r-', 'LineWidth', 1.6);
yline(K, 'k--', 'Label', 'Ea', 'FontSize', 10);
hold off;
xlabel('Periods since t = 50');
ylabel('Wealth');
legend('95% band (bottom 10%)', 'Mean (bottom 10%)', 'Aggregate wealth', 'Location', 'northwest', 'FontSize', 10);
exportgraphics(gcf, '../figures/fig_h_bottom10.png', 'Resolution', 300);

% Comments:
% The model's wealth mobility from the bottom is gradual, not fast.
% The poor still converge toward the mean eventually, even though gradual
% catch-up is aligns with data. In data: persistence at the bottom is 
% stronger, partly because of permanent differences between households 
% that this model doesn't have. Z mean-reverts and is irreducible, i.e.
% every household's long-run wealth distribution is the same whatever 
% its starting point.

%% part j. Lorenz curve and Gini (pooled cross-section, t >= 50)
% lorenz curve (equal weight 1/M per observation)
vapool = ma(:,tcan+1:end);
vapool = sort(vapool(:));                   % pooled wealth, ascending
pnumobs = numel(vapool);                    % M = (T-49)*N
vpop = [0; (1:pnumobs)'/pnumobs];           % cumulative population share
vlor = [0; cumsum(vapool)/sum(vapool)];     % cumulative wealth share
gini = 1 - 2*trapz(vpop, vlor);             % gini = 1 - 2*(area under lorenz)

% compare: histogram (stationary distribution, RCE) lorenz and gini
vwealthh = vgrida'.*marga;                  % wealth held at each grid point
vpoprce = [0; cumsum(marga)];
vlorrce = [0; cumsum(vwealthh)/sum(vwealthh)];
ginirce = 1 - 2*trapz(vpoprce, vlorrce);

% print results
fprintf('gini (simulation): %.4f | gini (histogram): %.4f \n', gini, ginirce);

% plot
figure;
plot(vpop, vlor, 'LineWidth', 1.6);
hold on;
plot(vpoprce, vlorrce, '--', 'LineWidth', 1.6);
plot([0 1], [0 1], 'k-');                   % perfect equality
hold off;
axis square;
xlabel('Cumulative population share');
ylabel('Cumulative wealth share');
legend('Simulation (pooled)', 'Histogram', '45° line', 'Location', 'northwest', 'FontSize', 10);
exportgraphics(gcf, '../figures/fig_j_lorenz.png', 'Resolution', 300);

% store previous results for comparison with part k
ginirce_pre = ginirce;
vpoprce_pre = vpoprce;
vlorrce_pre = vlorrce;

% Comments:
% Gini understates typical US wealth inequality (0.8-0.9) 
% because the model lacks permanent heterogeneity, return risk and aggregate risk.

%% part k. Productivity Shock and Wealth Inequality
pbeta = 0.96;
palpha = 0.36;
pdelta = 0.08;
pA = 1.20; % 20% increase in productivity (TFP) 
pnumgrida = 200;
pnumgridz = 7;
pmina = 0;
pmaxa = 300;
pcurve = 2;
solvege;

% post-shock wealth gini and lorenz
vwealthh_post = vgrida'.*marga;                  % wealth held at each grid point
vpoprce_post = [0; cumsum(marga)];
vlorrce_post = [0; cumsum(vwealthh_post)/sum(vwealthh_post)];
ginirce_post = 1 - 2*trapz(vpoprce_post, vlorrce_post);

% compare TFP effect on inequality measures
fprintf('gini (pre-shock): %.4f | gini (post-shock): %.4f \n', ginirce_pre, ginirce_post);
figure;
plot(vpoprce_pre, vlorrce_pre, 'LineWidth', 1.6);
hold on;
plot(vpoprce_post, vlorrce_post, '--', 'LineWidth', 1.6);
plot([0 1], [0 1], 'k-');                   % perfect equality
hold off;
axis square;
xlabel('Cumulative population share');
ylabel('Cumulative wealth share');
legend('Pre-shock', 'Post-shock', '45° line', 'Location', 'northwest', 'FontSize', 10);
exportgraphics(gcf, '../figures/fig_k_lorenz_tfp.png', 'Resolution', 300);

% Comments:
% Permanent TFP shock scales the economy uniformly because the standard Aiyagari-Beweley 
% economies features homothetic preferences and a production function with CRTS.