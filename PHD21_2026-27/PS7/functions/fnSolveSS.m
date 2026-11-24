function res = fnSolveSS(par, opt)
%% Solve Krusell and Smith (1998) SRCE with endogenous labour and wealth adjustment costs
% inputs:
%   par : parameters (palpha, pbeta, pdelta, pfrisch, peta, pmu, pA, prhoz, psigz,
%         pnumgrida, pnumgridz, pmina, pmaxa, pcurve)
%   opt : macros (verbose1, verbose2, verbose3, iterdisp, vecdist)
% output:
%   res : equilibrium objects (aggregates, prices, policies, distribution), pA and peta
%
% notes:
%   log utility (sigma = 1) and adjustment cost psi(a',a) = (pmu/2)*((a'-a)/a)^2*a
%   psi_1(a',a) = d psi/d a' = pmu*(a'-a)/a
%   psi_2(a',a) = d psi/d a  = -(pmu/2)*((a'/a)^2 - 1)
%   next-period labour solved in closed form, valid for pfrisch = 1 only
%
%% Model Fundamentals
%==============================
% unpack
%==============================
% parameters
palpha    = par.palpha;
pbeta     = par.pbeta;
pdelta    = par.pdelta;
pfrisch   = par.pfrisch;
peta      = par.peta;
pmu       = par.pmu;
pA        = par.pA;
prhoz     = par.prhoz;
psigz     = par.psigz;
pnumgrida = par.pnumgrida;
pnumgridz = par.pnumgridz;
pmina     = par.pmina;
pmaxa     = par.pmaxa;
pcurve    = par.pcurve;

% macros
verbose1  = opt.verbose1;
verbose2  = opt.verbose2;
verbose3  = opt.verbose3;
iterdisp  = opt.iterdisp;
vecdist   = opt.vecdist;

%==============================
% productivity distribution
%==============================
pmuz = 0.0;
[mtransz, vgridz] = fnTauchen(prhoz, pmuz, psigz^2, pnumgridz, 3.0); % psigz is innovation sd
vgridz = exp(vgridz);

% invariant distribution
vpiz = ones(size(mtransz, 1), 1) / size(mtransz, 1);
for i = 1:1000
    vpi = mtransz' * vpiz;
    vpiz = vpi;
end

%==============================
% wealth grid
%==============================
x = linspace(0, 0.5, pnumgrida);
y = x.^pcurve / max(x.^pcurve); % normalised to [0,1], clustered near 0 (Maliar, Maliar and Valli, 2010)
vgrida = pmina + (pmaxa - pmina)*y;

%% Numerical Setup
%==============================
% equilibrium objects
%==============================
mpolc = repmat(0.01*vgrida',1,pnumgridz);
mpoln = zeros(pnumgrida,pnumgridz);
mpolaprime = ones(pnumgrida,pnumgridz)*pmina;
mpolaprime_new = zeros(pnumgrida,pnumgridz);
mlambda = zeros(pnumgrida,pnumgridz);
mlambda_new = zeros(pnumgrida,pnumgridz);

%==============================
% auxiliary objects
%==============================
mgridz = repmat(vgridz',pnumgrida,1);
mgrida = repmat(vgrida',1,pnumgridz);
mgridiz = repmat(1:pnumgridz,pnumgrida,1); % z index

%==============================
% numerical parameters
%==============================
tolge = 1e-12;
toldist = 1e-10;
maxiter = 20000;
wt1 = 0.9000;
wt2 = 0.9000;
wt3 = 0.9000;
wt4 = 0.9000;

%==============================
% initial guess
%==============================
K = 7.50;
supplyL = 0.33;
mcurrdist = ones(pnumgrida,pnumgridz) / (pnumgrida*pnumgridz);

%% Numerical Solution
%==============================
% warm start
%==============================
if verbose3 && isfile('../solutions/WIP_ss.mat')
    load('../solutions/WIP_ss.mat','K','supplyL','mcurrdist','mpolaprime','mlambda'); % overwrites initial guess
end
hfig = figure('Visible', verbose2); % hidden unless plotting interim progress

%===============
% outer ge loop
%===============
errge = 10;
iterge = 1;
timerge = tic;
while errge > tolge && iterge <= maxiter

%===============
% prices
%===============
% given aggregates, prices known
K2L = K/supplyL;
r = palpha*pA*K2L^(palpha-1) - pdelta;
w = (1-palpha)*pA*K2L^palpha;

%===============
% inner loop (pfi)
%===============
% pfi step updates with outer ge loop
% improves stability avoids inner loop convergence issues

% future state/prices
rprime = r;
wprime = w;

% expectations (rhs euler, rational expectations)
mexp = 0;
for izp = 1:pnumgridz
    % future savings and marginal adjustment cost psi_2(a'',a')
    aprimeprime = interp1(vgrida', mpolaprime(:,izp), mpolaprime, "linear", "extrap");
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
c = 1./(mexp + mlambda);
c(c<1e-10) = 1e-10;
n = (w*mgridz./(peta*c)).^pfrisch;
n(n<0) = 0;
n(n>1) = 1;
madj = (pmu/2)*((mpolaprime-mgrida)./mgrida).^2.*mgrida;
mpolaprime_new = w*mgridz.*n + (1+r)*mgrida - c - madj;
mlambda_new = 1./(w*mgridz.*n + (1+r)*mgrida - mpolaprime - madj) - mexp;
mlambda_new(mlambda_new<0) = 0; % economically relevant sign

% constraints (kkt)
mlambda_new(mpolaprime_new>pmina) = 0; % constraint slack
mpolaprime_new(mpolaprime_new<=pmina) = pmina;

% update consumption
mpolc = c; % jump-update
mpoln = n;

%%
%===============
% stationary distribution
%===============
% histogram evolving method - non-stochastic
iterdist = 1;
errdist = 10;

% vectorised (sparse transition matrix)
if vecdist
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
    vcurrdist = mcurrdist(:);
    while errdist > toldist
        vnextdist = mT'*vcurrdist;
        errdist = max(abs(vnextdist-vcurrdist));
        vcurrdist = vnextdist;
        iterdist = iterdist + 1;
    end
    mcurrdist = reshape(vcurrdist,pnumgrida,pnumgridz);
% non-vectorised
else
    while errdist > toldist

        % next distribution (point-wise)
        mnextdist = zeros(pnumgrida,pnumgridz); % reset
        for iz = 1:pnumgridz
        for ia = 1:pnumgrida
            nexta = mpolaprime_new(ia,iz);
            [lo,hi,wlo,whi] = fnInterp1dGrid(nexta,vgrida,pnumgrida); % interpolate aprime on grid
            mass = mcurrdist(ia,iz);

            for izp = 1:pnumgridz
                mnextdist(lo,izp) = mnextdist(lo,izp) + mass*mtransz(iz,izp)*wlo;
                mnextdist(hi,izp) = mnextdist(hi,izp) + mass*mtransz(iz,izp)*whi;
            end
        end
        end

        % update
        errdist = max(abs(mnextdist-mcurrdist),[],"all");
        mcurrdist = mnextdist;
        iterdist = iterdist + 1;
    end
end

%%
%===============
% aggregates (mcc)
%===============
marga = sum(mcurrdist,2);
Knew = vgrida*marga;
supplyLnew = sum(mcurrdist.*mpoln.*mgridz,"all"); % effective labour
hours = sum(mcurrdist.*mpoln,"all");              % average hours
lambda = sum(mcurrdist.*mlambda,"all");
lambda_new = sum(mcurrdist.*mlambda_new,"all");

%===============
% update (ge)
%===============
% error
errge = mean(abs(...
    [Knew-K;...
    supplyLnew-supplyL;...
    lambda_new-lambda].^2),...
    'all');

% update
K          = wt1*K           + (1-wt1)*Knew;
supplyL    = wt2*supplyL     + (1-wt2)*supplyLnew;
mpolaprime = wt3.*mpolaprime + (1-wt3).*mpolaprime_new;
mlambda    = wt4.*mlambda    + (1-wt4).*mlambda_new;

%===============
% interim report
%===============
timelapse = toc(timerge);
if verbose1 && mod(iterge, iterdisp)==0 || iterge == 1 || errge <= tolge

    % prepare variables
    Y       = pA*K^palpha*supplyL^(1-palpha);
    K2Y     = K/Y;
    I2Y     = pdelta*K/Y;

    % print
    fprintf(' \n');
    fprintf('** Market Clearing Results ** \n');
    fprintf('iter %d (%.2f mins): error = %.15f \n', iterge, timelapse/60, errge);
    fprintf('K = %.4f | L = %.4f | hours = %.4f | peta = %.4f \n', K, supplyL, hours, peta);

    % save (mid)
    dir = '../solutions/WIP_ss.mat';
    save(dir);

    % plot
    if verbose2
        set(0,'CurrentFigure',hfig);
        plot(vgrida,marga,'LineWidth',2);
        drawnow;
    end
elseif mod(iterge, iterdisp)==0
    dir = '../solutions/WIP_ss.mat';
    save(dir);
end
iterge = iterge + 1;
end

%==============================
% final aggregates
%==============================
Y       = pA*K^palpha*supplyL^(1-palpha);
I       = pdelta*K;
C       = sum(mcurrdist.*mpolc,"all");
madj    = (pmu/2)*((mpolaprime_new-mgrida)./mgrida).^2.*mgrida;
ADJ     = sum(mcurrdist.*madj,"all");   % aggregate adjustment cost
K2Y     = K/Y;
I2Y     = I/Y;
C2Y     = C/Y;

%==============================
% save (final)
%==============================
dir = '../solutions/ss.mat';
save(dir);

%==============================
% print result
%==============================
fprintf('\n**** SRCE Results (A = %.2f) ****\n', pA);
fprintf('iter %d (%.2f mins): error = %.15f \n', iterge-1, timelapse/60, errge);
fprintf('K = %.4f | L = %.4f | hours = %.4f | r = %.4f | w = %.4f \n', K, supplyL, hours, r, w);
fprintf('K/Y = %.4f | I/Y = %.4f | C/Y = %.4f | ADJ/Y = %.6f \n', K2Y, I2Y, C2Y, ADJ/Y);
fprintf('peta = %.6f \n', peta);
fprintf('resource check: Y-C-I-ADJ = %.2e \n', Y-C-I-ADJ);

set(0,'CurrentFigure',hfig);
plot(vgrida,marga,'LineWidth',2);
ylabel('Density');
xlabel('Wealth');
exportgraphics(gcf, sprintf('../figures/ss_wealth_distn_A%.2f.png', pA), 'Resolution', 300);

%==============================
% results
%==============================
% parameters
res.pA          = pA;
res.peta        = peta;

% aggregates and prices
res.K           = K;
res.supplyL     = supplyL;
res.hours       = hours;
res.Y           = Y;
res.C           = C;
res.I           = I;
res.ADJ         = ADJ;
res.r           = r;
res.w           = w;
res.K2Y         = K2Y;
res.I2Y         = I2Y;

% grids and exogenous process
res.vgrida      = vgrida;
res.vgridz      = vgridz;
res.mtransz     = mtransz;
res.vpiz        = vpiz;

% policies and multipliers
res.mpolaprime  = mpolaprime_new;
res.mpolc       = mpolc;
res.mpoln       = mpoln;
res.mlambda     = mlambda_new;

% distribution
res.mcurrdist   = mcurrdist;
res.marga       = marga;

% convergence
res.errge       = errge;
res.iterge      = iterge-1;

end
