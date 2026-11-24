function res = fnSolveTCE(par, srceold, srcenew, exogtranspath, opt)
%% Solve Krusell and Smith (1998) TCE with endogenous labour and wealth adjustment costs
% inputs:
%   par           : parameters (palpha, pbeta, pdelta, pfrisch, peta, pmu, pnumgrida, pnumgridz,
%                   pmina, pathlength) and grids from fnSolveSS (vgrida, vgridz, mtransz)
%   srceold       : initial srce (start distribution, K)
%   srcenew       : terminal srce (end policies, K, L)
%   exogtranspath : perfect foresight tfp path (pathlength x 1)
%   opt           : macros (verbose1, verbose2, iterdisp)
% output:
%   res : transition paths (aggregates, prices, policies, distribution)
%
% notes:
%   log utility (sigma = 1) and adjustment cost psi(a',a) = (pmu/2)*((a'-a)/a)^2*a
%   next-period labour solved in closed form, valid for pfrisch = 1 only
%   guessed paths: tK (predetermined at t=1) and tsupplyL (jumps at t=1)
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
pathlength  = par.pathlength;

% grids and exogenous process (from fnSolveSS)
vgrida      = par.vgrida;
vgridz      = par.vgridz;
mtransz     = par.mtransz;

% boundary conditions
startdist   = srceold.mcurrdist;
Kstart      = srceold.K;
Kend        = srcenew.K;
Lend        = srcenew.supplyL;

% macros
verbose1  = opt.verbose1;
verbose2  = opt.verbose2;
iterdisp  = opt.iterdisp;

%==============================
% aggregate shock (TFP) path
%==============================
tA = exogtranspath;

%% Numerical Setup
%==============================
% equilibrium objects
%==============================
tmpolc          = repmat(srcenew.mpolc,1,1,pathlength);
tmpolc_new      = repmat(srcenew.mpolc,1,1,pathlength);
tmpoln_new      = repmat(srcenew.mpoln,1,1,pathlength);
tmpolaprime     = repmat(srcenew.mpolaprime,1,1,pathlength);
tmpolaprime_new = repmat(srcenew.mpolaprime,1,1,pathlength);
tmlambda        = repmat(srcenew.mlambda,1,1,pathlength);
tmlambda_new    = repmat(srcenew.mlambda,1,1,pathlength);

%==============================
% auxiliary objects
%==============================
mgridz = repmat(vgridz',pnumgrida,1);
mgrida = repmat(vgrida',1,pnumgridz);
mgridiz = repmat(1:pnumgridz,pnumgrida,1); % z index
mzoffset = (mgridiz-1)*pnumgrida;          % linear index offset for z columns

%==============================
% numerical parameters
%==============================
toltce = 1e-6;
% toldist = 1e-10;
maxiter = 5000;
wt1 = 0.8000;
wt2 = 0.8000;
wt3 = 0.8000;

%==============================
% initial guess
%==============================
tK = Kend * ones(pathlength,1);
tK(1) = Kstart; % predetermined K at t=1
tsupplyL = Lend * ones(pathlength,1);

%% Numerical Solution
%===============
% outer tce loop
%===============
hfig = figure('Visible', verbose2); % hidden unless plotting interim progress

iter = 1;
err = 10;
timertce = tic;
while err > toltce && iter <= maxiter

%===============
% update price path
%===============
tK2L = tK./tsupplyL;
tr = palpha.*tA.*tK2L.^(palpha-1) - pdelta;
tw = (1-palpha).*tA.*tK2L.^palpha;

%%
%===============
% backward solution
%===============
% update policy path backwards
for itrans = pathlength-1:-1:1

% current prices
w = tw(itrans);
r = tr(itrans);

% future prices
wprime = tw(itrans+1);
rprime = tr(itrans+1);

% current guess
mpolaprime = tmpolaprime(:,:,itrans);

% expectations (rhs euler, perfect foresight in aggregates)
mexp = 0;
for izp = 1:pnumgridz
    % future savings and marginal adjustment cost psi_2(a'',a')
    aprimeprime = interp1(vgrida', tmpolaprime(:,izp,itrans+1), mpolaprime, "linear", "extrap");
    psi2 = -(pmu/2)*((aprimeprime./mpolaprime).^2 - 1);
    adjprime = (pmu/2)*((aprimeprime-mpolaprime)./mpolaprime).^2.*mpolaprime;

    % future labour (closed form intratemporal foc, pfrisch = 1)
    mprime = ((1+rprime)*mpolaprime - adjprime - aprimeprime)/(wprime*vgridz(izp));
    nprime = (-peta*mprime + sqrt((peta*mprime).^2 + 4*peta))/(2*peta);
    nprime(nprime>1) = 1;

    % future consumption
    cprime = wprime*vgridz(izp)*nprime + (1+rprime)*mpolaprime - aprimeprime - adjprime;
    cprime(cprime<1e-10) = 1e-10;
    muprime = 1./cprime;

    % update expectations
    mexp = mexp + mtransz(:,izp)'.*(1+rprime-psi2).*muprime;
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
tmpolc_new(:,:,itrans) = c; % jump-update
tmpoln_new(:,:,itrans) = n;
tmpolaprime_new(:,:,itrans) = mpolaprime_newtemp;
tmlambda_new(:,:,itrans) = mlambda_newtemp;
end
tmpolc = tmpolc_new; % jump update

%%
%===============
% forward simulation
%===============
% non-stochastic histogram method (vectorised)
% interpolate entire policy path on grid once per outer iteration
[tlo, thi, twlo, twhi] = fnInterp1dVec(tmpolaprime_new, vgrida);

% initialise
tcurrdist = zeros(pnumgrida,pnumgridz,pathlength);
tcurrdist(:,:,1) = startdist; % old srce distribution

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

%===============
% aggregates (mcc)
%===============
tK_new = squeeze(sum(tcurrdist.*mgrida,[1 2]));                   % capital supply
tsupplyL_new = squeeze(sum(tcurrdist.*tmpoln_new.*mgridz,[1 2])); % effective labour
thours = squeeze(sum(tcurrdist.*tmpoln_new,[1 2]));               % average hours

%===============
% update (tce)
%===============
% error
tlambda_err = sum(tcurrdist(:,:,1:pathlength-1) ...
    .*abs(tmlambda_new(:,:,1:pathlength-1) - tmlambda(:,:,1:pathlength-1)),'all');

err = mean(abs([ ...
    tK_new(2:pathlength) - tK(2:pathlength); ...
    tsupplyL_new - tsupplyL; ...
    tlambda_err].^2));

% update
tK          = wt1.*tK          + (1-wt1).*tK_new;
tsupplyL    = wt1.*tsupplyL    + (1-wt1).*tsupplyL_new;
tmpolaprime = wt2.*tmpolaprime + (1-wt2).*tmpolaprime_new;
tmlambda    = wt3.*tmlambda    + (1-wt3).*tmlambda_new;
tK(1)          = Kstart; % predetermined
tK(pathlength) = Kend;   % snap end to new srce
tsupplyL(pathlength) = Lend;

%===============
% interim report
%===============
timelapse = toc(timertce);
if verbose1 && mod(iter, iterdisp)==0 || iter == 1 || err <= toltce

    % print
    fprintf(' \n');
    fprintf('** TCE Results ** \n');
    fprintf('iter %d (%.2f mins): error = %.15f \n', iter, timelapse/60, err);
    fprintf('K end gap: %.8f   |  max |dK|: %.8f   |  max |dL|: %.8f \n', ...
        tK_new(pathlength)-Kend, max(abs(tK_new(2:pathlength) - tK(2:pathlength))), ...
        max(abs(tsupplyL_new - tsupplyL)));

    % plot
    if verbose2
        set(0,'CurrentFigure',hfig); % plot to own figure even if another is clicked
        subplot(1,2,1);
        plot(1:pathlength, tK, 'b-', 'LineWidth', 1.6);
        hold on;
        plot(1:pathlength, tK_new, 'r-.', 'LineWidth', 1.6);
        hold off;
        yline(Kend, 'k--','Label','end');
        yline(Kstart, 'k--','Label','start');
        xlabel('Period');
        ylabel('K');
        legend('Predicted', 'Realised', 'Location', 'east','FontSize',12);
        subplot(1,2,2);
        plot(1:pathlength, tsupplyL, 'b-', 'LineWidth', 1.6);
        hold on;
        plot(1:pathlength, tsupplyL_new, 'r-.', 'LineWidth', 1.6);
        hold off;
        yline(Lend, 'k--','Label','end');
        xlabel('Period');
        ylabel('L');
        drawnow;
    end
end

% dynamic weight updates
if iter>1000
    wt1 = 0.9500;
    wt2 = 0.9500;
    wt3 = 0.9500;
end
iter = iter + 1;
end % outer tce loop

%==============================
% transition allocations
%==============================
tY = tA.*tK.^palpha.*tsupplyL.^(1-palpha);
tC = squeeze(sum(tcurrdist.*tmpolc,[1 2]));
tI = [tK(2:pathlength) - (1-pdelta)*tK(1:pathlength-1); pdelta*Kend];
tmadj = (pmu/2)*((tmpolaprime_new-mgrida)./mgrida).^2.*mgrida;
tADJ = squeeze(sum(tcurrdist.*tmadj,[1 2]));

%==============================
% save (final)
%==============================
dir = '../solutions/tce.mat';
save(dir);

%==============================
% print result
%==============================
fprintf('\n**** TCE Results ****\n');
fprintf('iter %d (%.2f mins): error = %.15f \n', iter-1, timelapse/60, err);
fprintf('K: %.4f -> %.4f | L: %.4f -> %.4f \n', Kstart, Kend, tsupplyL(1), Lend);

%==============================
% results
%==============================
% aggregates and prices
res.tA          = tA;
res.tK          = tK;
res.tsupplyL    = tsupplyL;
res.thours      = thours;
res.tY          = tY;
res.tC          = tC;
res.tI          = tI;
res.tADJ        = tADJ;
res.tr          = tr;
res.tw          = tw;

% policies and distribution
res.tcurrdist   = tcurrdist;
res.tmpolaprime = tmpolaprime_new;
res.tmpolc      = tmpolc;
res.tmpoln      = tmpoln_new;
res.tmlambda    = tmlambda_new;

% convergence
res.err         = err;
res.iter        = iter-1;
res.converged   = err <= toltce;

end
