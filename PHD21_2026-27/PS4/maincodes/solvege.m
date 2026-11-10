%% Solve Aiyagari 1994 SRCE 
%  
% 
%% Model Fundamentals 
% %==============================
% % parameters (Aiyagari, 1994) 
% %==============================
% pbeta = 0.96;
% palpha = 0.36;
% pdelta = 0.08;
% pA = 1.0;
% pnumgrida = 100;
% pnumgridz = 7;
% pmina = 0;
% pmaxa = 300;

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

%% Numerical Setup 
%==============================
% equilibrium objects 
%==============================
mpolc = zeros(pnumgrida,pnumgridz);
mpolaprime = repmat(vgrida'.*0.2,1,pnumgridz);
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
tolge = 1e-10;
toldist = 1e-10;
maxiter = 5000;
wt1 = 0.9000;
wt2 = 0.8000;
wt3 = 0.8000;

%==============================
% initial guess 
%==============================
K = 3.40;
mcurrdist = ones(pnumgrida,pnumgridz) / (pnumgrida*pnumgridz);  

%% Numerical Solution
%==============================
% warm start
%==============================
if verbose3 && isfile('../solutions/WIP_ss.mat')
    load('../solutions/WIP_ss.mat','K','mcurrdist','mpolaprime','mlambda'); % overwrites initial guess
end
figure;

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
    % future consumption 
    aprimeprime = interp1(vgrida', mpolaprime(:,izp), mpolaprime, "linear", "extrap");
    cprime = wprime*vgridz(izp) + (1+rprime)*mpolaprime - aprimeprime;
    cprime(cprime<1e-10) = 1e-10;
    muprime = 1./cprime;

    % update expectations 
    mexp = mexp + mtransz(:,izp)'.*(1+rprime).*muprime;
end
mexp = pbeta*mexp;

% optimal intratemporal policies 
c = 1./(mexp + mlambda);
c(c<1e-10) = 1e-10;
mpolaprime_new = w*mgridz + (1+r)*mgrida - c;
mlambda_new = 1./(w*mgridz + (1+r)*mgrida - mpolaprime) - mexp;
mlambda_new(mlambda_new<0) = 0; % economically relevant sign

% constraints (nnc, kkt) 
mlambda_new(mpolaprime_new>pmina) = 0; % constraint slack 
c = c.*(mpolaprime_new>=pmina) + (c - (pmina - mpolaprime_new)).*(mpolaprime_new<pmina);
c(c<1e-10) = 1e-10;
mpolaprime_new(mpolaprime_new<pmina) = pmina;

% update consumption 
% mpolc = wt2*mpolc + (1-wt2)*c; % damped 
mpolc = c; % jump-update 

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

%===============
% aggregates (mcc)
%===============
marga = sum(mcurrdist,2);
Knew = vgrida*marga;
% Knew = sum(mcurrdist.*mpolaprime_new,"all");
supplyLnew = sum(mcurrdist.*mgridz,"all");
lambda = sum(mcurrdist.*mlambda,"all");
lambda_new = sum(mcurrdist.*mlambda_new,"all");
aprime_err = sum(mcurrdist.*abs(mpolaprime_new-mpolaprime),"all");

%===============
% update (ge)
%===============
% error
errge = mean(abs(...
    [Knew-K;...
    aprime_err;...
    lambda_new-lambda].^2),...
    'all');

% update
K          = wt1*K           + (1-wt1)*Knew;
mpolaprime = wt2.*mpolaprime + (1-wt2).*mpolaprime_new;
mlambda    = wt3.*mlambda    + (1-wt3).*mlambda_new;

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

    % save (mid)
    dir = '../solutions/WIP_ss.mat';
    save(dir);

    % plot
    if verbose2
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
% save (final)
%==============================
dir = '../solutions/ss.mat';
save(dir);

%==============================
% print result
%==============================
fprintf('\n**** SRCE Results ****\n');
fprintf('iter %d (%.2f mins): error = %.15f \n', iterge-1, timelapse/60, errge);
fprintf('K = %.4f, supplyL = %.4f, r = %.4f, w = %.4f \n', K, supplyL, r, w);
fprintf('K/Y = %.4f, I/Y = %.4f \n', K2Y, I2Y);

plot(vgrida,marga,'LineWidth',2);
ylabel('Density');
xlabel('Wealth');
exportgraphics(gcf, '../figures/ss_wealth_distn.png', 'Resolution', 300);