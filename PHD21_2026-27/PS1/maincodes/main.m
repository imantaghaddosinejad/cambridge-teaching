%% Cambridge PHD21 Computational Methods 2026-27
% Problem Set 1
% Lecturer: Dr. Hanbaek Lee (https://sites.google.com/view/hanbaeklee/)
% TA: Iman Taghaddosinejad (https://github.com/imantaghaddosinejad)
%%
%==============================
% housekeeping 
%==============================
clear;
clc;
addpath('../functions');
addpath('../figures');

%==============================
% parameters 
%==============================
pbeta = 0.96;
ptau = 0.15;
palpha = 0.30;
pa = 1;
A = 1;
r = 0.04;

%==============================
% productivity distribution 
%==============================
% log z ~ iid N(pmuz,psiggz^2)
pnumgridz = 30;
psiggz = 0.20;
pmuz = 0.0;
prhoz = 0.0;
[mtransz, vgridz] = fnTauchen(prhoz, pmuz, psiggz^2, pnumgridz, 3.0); 
vgridz = exp(vgridz); % lognormal productivity grid

% invariant distribution 
vpiz = ones(size(mtransz, 1), 1) / size(mtransz, 1);
for i = 1:1000
    vpi = mtransz' * vpiz;
    vpiz = vpi;
end

% grid properties  
fprintf('sum pi: %.6f   |  E[log z]: %.6f   |  sd(log z): %.4f   |  E[z]: %.4f \n', ...
    sum(vpiz), vpiz'*log(vgridz), sqrt(vpiz'*log(vgridz).^2), vpiz'*vgridz);

%%
%==============================
% individual labor supply curve 
%==============================
pnumgridw=30;
pfrisch=1.0;
peta = 3.64;
T = 0.058;
pb = 0.04;

vw = linspace(0.05, 4.0, pnumgridw);
mn = zeros(pnumgridw, pnumgridz);
for iw = 1:pnumgridw
    w = vw(iw);
    for iz = 1:pnumgridz
        z=vgridz(iz);
        mn(iw,iz) = intensive_labor_supply(pbeta, pfrisch, peta, ptau, pa, z, w, r, T);
    end
end

% plot
figure;
hold on;
for iz = 2:pnumgridz-1
    h = plot(vw, mn(:,iz), 'Color', [0.75 0.75 0.75], 'LineWidth', 0.8);
    if iz > 2
        h.HandleVisibility = 'off';     % one legend entry for all interior z
    else
        h.DisplayName = 'interior z';
    end
end
plot(vw, mn(:,1), 'b-', 'LineWidth', 2, ...
    'DisplayName', sprintf('lowest z = %.2f', vgridz(1)));
plot(vw, mn(:,end), 'r-', 'LineWidth', 2, ...
    'DisplayName', sprintf('highest z = %.2f', vgridz(end)));
hold off;
xlabel('wage (w)');
ylabel('hours (n)');
title('individual labour supply curve');
legend('show', 'Location', 'best');
saveas(gcf, '../figures/individual_labor_supply_curve.png');

%==============================
% aggregate labor supply curve 
%==============================
% compute extensive margin for each (w,z) pair 
mzw = vw'.*vgridz';
R   = 1 + (1-ptau)*r;

% worker
mc_lab = ((1-ptau)*mzw.*mn + pa*R + T)/(1+pbeta);
mc_lab(mc_lab<=1e-10) = 1e-10;
mpolaprime_lab = pbeta*mc_lab;
mW = fnW(pbeta, pfrisch, peta, mc_lab, mn, mpolaprime_lab);

% non-worker: independent of z and w
c_nonlab = (pb + pa*R + T)/(1+pbeta);
c_nonlab(c_nonlab<=1e-10) = 1e-10;
aprime_nonlab = pbeta*c_nonlab;
vN = fnN(pbeta, c_nonlab, aprime_nonlab);
mR = max(mW, vN);

% extensive margin
mwork = mW > vN;                          

% aggregate labor supply curve and employment at each w
vLs  = sum(mn .* mwork .* vgridz' .* vpiz',2); % effective labour
% vemp = mwork*vpiz; % employment rate

% aggregate labor demand curve 
vLd = pa*((1-palpha)*A./vw).^(1/palpha);

% plot labor AS vs AD curve 
figure;
hold on;
plot(vw, vLs, 'b-', 'LineWidth', 2, 'DisplayName', 'labour supply');
plot(vw, vLd, 'r-', 'LineWidth', 2, 'DisplayName', 'labour demand');
hold off;
ylim([0, 1.5*max(vLs)]);
xlabel('wage (w)');
ylabel('effective labour (L)');
title('aggregate labour supply and demand');
legend('show', 'Location', 'best');
saveas(gcf, '../figures/aggregate_labor_supply_and_demand.png');
%%
%==============================
% GE solution 
%==============================
% equilibrium objects 
vn_lab = zeros(pnumgridz, 1);
vc_lab = zeros(pnumgridz, 1);
vaprime_lab = zeros(pnumgridz, 1);

% initial guess 
L = 0.33;
T = 0.058;

% updating weights 
wt1 = 0.9000;
wt2 = 0.9000;

% peta = 3.64;
pb = 0.086;
% psiggz = 0.20;

% RCE loop
tol_ge = 1e-10;
err_ge = 10;
pnumiter = 1;
tic;
while err_ge > tol_ge

% prices given aggregates  
w = (1-palpha)*A*(pa/L)^(palpha);

%====================
% policies 
%====================     
% workers 
for iz = 1:pnumgridz 
    z = vgridz(iz);
    n = intensive_labor_supply(pbeta, pfrisch, peta, ptau, pa, z, w, r, T);
    n(n<0) = 0;
    n(n>1) = 1;        
    c = ((1-ptau)*z*w*n + pa*R + T)/(1 + pbeta);
    c(c<1e-10) = 1e-10;
    vaprime_lab(iz) = pbeta*c;
    vn_lab(iz) = n;
    vc_lab(iz) = c;
end

% non-workers    
vc_nonlab = ones(pnumgridz, 1) .* ((pb + pa*R + T)/(1+pbeta));
vc_nonlab(vc_nonlab<=1e-10) = 1e-10;
vaprime_nonlab = pbeta*vc_nonlab;

% value functions and extensive margin 
vW = fnW(pbeta, pfrisch, peta, vc_lab, vn_lab, vaprime_lab);
vN = fnN(pbeta, vc_nonlab, vaprime_nonlab);
vR = max(vW, vN);
vwork = vW > vN; % extensive margin 

%====================
% aggregates (mcc)
%====================
L_new = sum(vn_lab.*vwork.*vgridz.*vpiz);
T_new = ptau*(w*L_new + r*pa) - pb*sum((~vwork).*vpiz);

%====================
% update 
%====================
% ge error 
err_ge = max(abs([L_new-L; T_new-T].^2));

% convex updating 
L = wt1*L + (1-wt1)*L_new;
T = wt2*T + (1-wt2)*T_new;

%====================
% interim report
%====================     
if mod(pnumiter, 5) == 0 || pnumiter == 1 || err_ge < tol_ge
    fprintf('\n*** Market Clearing Report ***\n');
    fprintf('iter = %d (%.2fs): err_ge = %.12f \n', pnumiter, toc/60, err_ge)
    fprintf('L = %.4f | T = %.4f \n', L, T);
end
pnumiter = 1 + pnumiter;
end

%====================
% plot value functions
%====================     
figure;
hold on;
plot(vgridz, vW, 'b-o', 'LineWidth', 2, 'MarkerSize', 4, 'DisplayName', 'worker (W)');
plot(vgridz, vN, 'r-o', 'LineWidth', 2, 'MarkerSize', 4, 'DisplayName', 'non-worker (N)');
plot(vgridz, vR, 'k--', 'LineWidth', 2, 'DisplayName', 'R');
hold off;
xline(vgridz(sum(vW<=vN)), 'k--', 'LineWidth', 1.5, ...
    'HandleVisibility', 'off', 'Label', 'z^*', ...
    'LabelOrientation','horizontal', 'LabelVerticalAlignment','bottom');
xlabel('productivity (z)');
ylabel('value function');
title('value functions');
legend('show', 'Location', 'east');
axis tight;
%%
%==============================
% GE: targeted moments
%==============================
par.pbeta      = pbeta;
par.palpha     = palpha;
par.ptau       = ptau;
par.pa         = pa;
par.A          = A;
par.r          = r;
par.pfrisch    = 1.0;
par.prhoz      = prhoz;
par.pmuz       = pmuz;
par.pnumgridz  = pnumgridz;
par.tol_ge     = 1e-10;
par.maxiter_ge = 2000;
par.wt1        = 0.50;
par.wt2        = 0.50;
par.L0         = 0.33; % initial guess for L    
par.T0         = 0.058; % initial guess for T

% targets and weights
vtarget = [...
    0.330, ...  % average hours of workers
    0.060, ...  % unemployment rate
    0.250, ...  % b / avg wage income of workers
    0.700];     % std/mean ratio of wage income of workers
mweight = diag([1, 1, 0, 1]);

% estimate
x0 = log([peta, pb, psiggz]);
options = optimset('Display', 'iter', 'TolX', 1e-8, 'TolFun', 1e-12, 'MaxFunEvals', 5000);
[xhat, fval] = fminsearch(@(x) fnLoss(x, par, vtarget, mweight), x0, options);
peta   = exp(xhat(1));
pb     = exp(xhat(2));
psiggz = exp(xhat(3));

% report calibration
fprintf('\n*** Calibration Report ***\n');
fprintf('peta = %.4f | pb = %.4f | psiggz = %.4f \n', peta, pb, psiggz);

% equilibrium at the estimates
par.peta   = peta;
par.pb     = pb;
par.psiggz = psiggz;
[mtransz, vlogz] = fnTauchen(prhoz, pmuz, psiggz^2, pnumgridz, 3.0);
vgridz = exp(vlogz);
vpiz = mtransz(1,:)';
[L, T, w, vn_lab, vc_lab, vwork, flag] = fnSolveGE(par, vgridz, vpiz, par.L0, par.T0);

% moments and moment errors
vM = fnMom(w, vn_lab, vwork, vgridz, vpiz, pb);
verr = 100 * (vM - vtarget)./vtarget;
vnames = {'avg hours (workers)', 'unemployment rate', 'b / avg wage income', 'std/mean wage income'};

% print
fprintf('GE converged: %d | w = %.4f | L = %.4f | T = %.4f \n', flag, w, L, T);
fprintf('%-24s %10s %10s %10s %8s \n', 'moment', 'target', 'model', 'error (%)', 'weight');
for im = 1:numel(vtarget)
    fprintf('%-24s %10.4f %10.4f %10.4f %8.2f \n', ...
        vnames{im}, vtarget(im), vM(im), verr(im), mweight(im,im));
end
fprintf('loss = %.3e | fminsearch fval = %.3e \n', verr*mweight*verr', fval);
%%
%==============================
% comparative statics: TPF and wage
%==============================
pnumgridA = 10;
par.A = 1.00;
vflagA = zeros(pnumgridA,1);
vw = zeros(pnumgridA,1);
vgridA = linspace(par.A*0.2, par.A*1.8, pnumgridA); 

% RCE loop over TFP 
L0 = L;
T0 = T;
for iA = 1:pnumgridA 
    par.A = vgridA(iA);    
    [L0, T0, vw(iA), ~, ~, ~, vflagA(iA)] = fnSolveGE(par, vgridz, vpiz, L0, T0); % warm start (L,T) 
end
par.A = 1.00; % reset TFP 

% plot
figure;
plot(vgridA, vw, 'b-o', 'LineWidth', 2, 'MarkerSize', 4);
xlabel('TFP (A)');
ylabel('wage (w)');
title('comparative statics: GE wage in A');
%%
%==============================
% counterfactual exercise: tax hike and inequality 
%==============================
vtau = [0.15, 0.16]; % tax rate counterfactuals 
mineq = zeros(numel(vtau), 4); % inequality matrix: (gini, varlog) for disposable and market inc

% tax experiment
for itau = 1:numel(vtau)
    par.ptau = vtau(itau);
    [L, T, w, vn_lab, vc_lab, vwork] = fnSolveGE(par, vgridz, vpiz, L, T);  

    % income and mass - non-workers stacked at the end
    vy_disp = [...
                (1-par.ptau)*(w*vgridz.*vn_lab + par.r*par.pa) + T; ...
                par.pb + (1-par.ptau)*par.r*par.pa + T];
    vy_mkt  = [...
                w*vgridz.*vn_lab + par.r*par.pa; ...
                par.r*par.pa];
    vmass   = [...
                vpiz.*vwork; ...
                sum(vpiz.*(~vwork))];

    % compute inequality
    [mineq(itau,1), mineq(itau,2)] = fnIneq(vy_disp, vmass);
    [mineq(itau,3), mineq(itau,4)] = fnIneq(vy_mkt, vmass);

    % print
    fprintf('tau = %.3f | w = %.4f | T = %.4f | u = %.4f \n', par.ptau, w, T, 1-sum(vpiz.*vwork));
    fprintf('gini disp = %.4f | varlog disp = %.4f | gini mkt = %.4f | varlog mkt = %.4f \n', ...
        mineq(itau,:));
    fprintf('\n');
end
par.ptau = 0.15; % reset tax rate 
%%
%==============================
% counterfactual exercise: disutility of labor  
%==============================
veta = [par.peta, par.peta * 1.20]; % disutility of labor counterfactuals 
mineq = zeros(numel(veta), 4); % inequality matrix: (gini, varlog) for disposable and market inc

% tax experiment
for ieta = 1:numel(veta)
    par.peta = veta(ieta);
    [L, T, w, vn_lab, vc_lab, vwork] = fnSolveGE(par, vgridz, vpiz, L, T);  

    % income and mass - non-workers stacked at the end
    vy_disp = [...
                (1-par.ptau)*(w*vgridz.*vn_lab + par.r*par.pa) + T; ...
                par.pb + (1-par.ptau)*par.r*par.pa + T];
    vy_mkt  = [...
                w*vgridz.*vn_lab + par.r*par.pa; ...
                par.r*par.pa];
    vmass   = [...
                vpiz.*vwork; ...
                sum(vpiz.*(~vwork))];

    % compute inequality
    [mineq(itau,1), mineq(itau,2)] = fnIneq(vy_disp, vmass);
    [mineq(itau,3), mineq(itau,4)] = fnIneq(vy_mkt, vmass);

    % print
    fprintf('eta = %.4f | w = %.4f | T = %.4f | u = %.4f \n', par.peta, w, T, 1-sum(vpiz.*vwork));
    fprintf('gini disp = %.4f | varlog disp = %.4f | gini mkt = %.4f | varlog mkt = %.4f \n', ...
        mineq(itau,:));
    fprintf('\n');
end
par.peta = peta; % reset disutility parameter