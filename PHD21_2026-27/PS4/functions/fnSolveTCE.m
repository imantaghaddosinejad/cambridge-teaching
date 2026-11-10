function res = fnSolveTCE(par, srceold, srcenew, exogtranspath, opt)
%
%
%
%% Model Fundamentals
%==============================
% unpack
%==============================
% parameters (Aiyagari, 1994)
pbeta       = par.pbeta;
palpha      = par.palpha;
pdelta      = par.pdelta;
% pA          = par.pA;
pnumgrida   = par.pnumgrida;
pnumgridz   = par.pnumgridz;
pmina       = par.pmina;
pmaxa       = par.pmaxa;
pcurve      = par.pcurve;
pathlength  = par.pathlength;
startdist   = srceold.mcurrdist;
Kstart      = srceold.K;
Kend        = srcenew.K;

% macros
verbose1  = opt.verbose1;
verbose2  = opt.verbose2;
% verbose3  = opt.verbose3;
iterdisp  = opt.iterdisp;
% vecdist   = opt.vecdist;

%==============================
% productivity distribution 
%==============================
pmuz = 0.0;
prhoz = 0.90;
psiggz_uncond = 0.20; % sd logz 
psiggz = psiggz_uncond*sqrt(1-prhoz^2); % innovation sd  
[mtransz, vgridz] = fnTauchen(prhoz, pmuz, psiggz^2, pnumgridz, 3.0); 
vgridz = exp(vgridz); 

% invariant distribution 
vpiz = ones(size(mtransz, 1), 1) / size(mtransz, 1);
for i = 1:1000
    vpi = mtransz' * vpiz;
    vpiz = vpi;
end

% inelastic aggregate labor supply 
supplyL = vpiz'*vgridz;

%==============================
% wealth grid  
%==============================
% pcurve = 5;
x = linspace(0, 0.5, pnumgrida); 
y = x.^pcurve / max(x.^pcurve); % normalised to [0,1], clustered near 0 (Maliar, Maliar and Valli, 2010)          
vgrida = pmina + (pmaxa - pmina)*y; 

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

%==============================
% numerical parameters
%==============================
toltce = 1e-8;
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

%% Numerical Solution
%===============
% outer tce loop
%===============
figure;

iter = 1;
err = 10;
converged = false;
tic;
while err > toltce && iter <= maxiter

%===============
% update price path 
%===============
tK2L = tK/supplyL;
% tK2L(pathlength) = Kend/supplyL; % snap end of path to new steady-state 
tr = palpha.*tA.*tK2L.^(palpha-1)-pdelta;
tw = (1-palpha).*tA.*tK2L.^palpha;

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

% expected future value (perfect foresight) 
mexp = 0;
for izp = 1:pnumgridz
    % future consumption 
    aprimeprime = interp1(vgrida', tmpolaprime(:,izp,itrans+1), tmpolaprime(:,:,itrans), "linear", "extrap");
    cprime = wprime*vgridz(izp) + (1+rprime)*tmpolaprime(:,:,itrans) - aprimeprime;
    cprime(cprime<1e-10) = 1e-10;
    muprime = 1./cprime;

    % update expectations 
    mexp = mexp + mtransz(:,izp)'.*(1+rprime).*muprime;
end
mexp = pbeta*mexp;

% optimal intratemporal policies 
c = 1./(mexp + tmlambda(:,:,itrans));
c(c<1e-10) = 1e-10;
mpolaprime_newtemp = w*mgridz + (1+r)*mgrida - c;
mlambda_newtemp = 1./(w*mgridz + (1+r)*mgrida - tmpolaprime(:,:,itrans)) - mexp;
mlambda_newtemp(mlambda_newtemp<0) = 0; % economically relevant sign

% constraints (nnc, kkt) 
mlambda_newtemp(mpolaprime_newtemp>pmina) = 0; % constraint slack 
c = c.*(mpolaprime_newtemp>=pmina) + (c - (pmina - mpolaprime_newtemp)).*(mpolaprime_newtemp<pmina);
c(c<1e-10) = 1e-10;
mpolaprime_newtemp(mpolaprime_newtemp<pmina) = pmina;

% update 
tmpolc_new(:,:,itrans) = c; % jump-update 
tmpolaprime_new(:,:,itrans) = mpolaprime_newtemp;
tmlambda_new(:,:,itrans) = mlambda_newtemp;
end
% tmpolc = wt2*tmpolc + (1-wt2)*tmpolc_new; % damped 
tmpolc = tmpolc_new; % jump update 

%===============
% forward simulation 
%===============
% non-stochastic histogram method 
% interpolate entire policy path on grid once per outer iteration
[tlo, thi, twlo, twhi] = fnInterp1dVec(tmpolaprime_new, vgrida);
mzoffset = (mgridiz-1)*pnumgrida; % linear index offset for z columns
 
% initialise
tcurrdist = zeros(pnumgrida,pnumgridz,pathlength);
tcurrdist(:,:,1) = startdist; % old srce distribution
tK_new = zeros(pathlength,1);
tK_new(1) = Kstart; % predetermined

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
    
    % aggregates (mcc)
    tK_new(itrans+1) = sum(currdist.*tmpolaprime_new(:,:,itrans),'all');
end

%===============
% update (tce)
%===============
% error 
taprime_err = sum(tcurrdist(:,:,1:pathlength-1) ...
    .*abs(tmpolaprime_new(:,:,1:pathlength-1) - tmpolaprime(:,:,1:pathlength-1)),'all');
tlambda_err = sum(tcurrdist(:,:,1:pathlength-1) ...
    .*abs(tmlambda_new(:,:,1:pathlength-1) - tmlambda(:,:,1:pathlength-1)),'all');

err = mean(abs([ ...
    tK(2:pathlength) - tK_new(2:pathlength); ...
    taprime_err; ...
    tlambda_err].^2));

% update 
tK          = wt1.*tK          + (1-wt1).*tK_new;
tmpolaprime = wt2.*tmpolaprime + (1-wt2).*tmpolaprime_new;
tmlambda    = wt3.*tmlambda    + (1-wt3).*tmlambda_new;
tK(1)       = Kstart; % predetermined
tK(pathlength) = Kend; % snap end to new srce 

%===============
% interim report
%===============
timelapse = toc;
if verbose1 && mod(iter, iterdisp)==0 || iter == 1 || err <= toltce

    % print
    fprintf(' \n');
    fprintf('** TCE Results ** \n');
    fprintf('iter %d (%.2f mins): error = %.15f \n', iter, timelapse/60, err);
    fprintf('K end gap: %.8f   |  max |dK|: %.8f \n', ...
        tK_new(pathlength)-Kend, max(abs(tK(2:pathlength) - tK_new(2:pathlength))));

    % plot
    if verbose2
        plot(1:pathlength, [Kstart;tK(1:end-1)], 'b-', 'LineWidth', 1.6);
        hold on;
        plot(1:pathlength, [Kstart;tK_new(1:end-1)], 'r-.', 'LineWidth', 1.6);
        hold off;
        yline(Kend, 'k--','Label','end');
        yline(Kstart, 'k--','Label','start')
        xlabel('Period');
        ylabel('K');
        legend('Predicted', 'Realised', 'Location', 'east','FontSize',12);
        drawnow;
    end
end

% dynamic weight updates 
if iter>2000 
    wt1 = 0.9500;
    wt2 = 0.9500;
    wt3 = 0.9500;
end
iter = iter + 1;
end % outer tce loop

%==============================
% results 
%==============================
if err <= toltce
converged=~converged;
end
res.tA          = tA;
res.tK          = tK;
res.tr          = tr;
res.tw          = tw;
res.tcurrdist   = tcurrdist;
res.tmpolaprime = tmpolaprime_new;
res.tmpolc      = tmpolc;
res.err         = err;
res.iter        = iter-1;
res.converged   = converged;
end









