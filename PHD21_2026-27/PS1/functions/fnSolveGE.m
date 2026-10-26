function [L, T, w, vn_lab, vc_lab, vwork, flag] = fnSolveGE(par, vgridz, vpiz, L, T)
% stationary rce: damped fixed point on (L,T) given parameters and z grid
R = 1 + (1-par.ptau)*par.r;
vn_lab = zeros(par.pnumgridz,1);
vc_lab = zeros(par.pnumgridz,1);
vaprime_lab = zeros(par.pnumgridz,1);

err_ge = 10;
iter_ge = 1;
while err_ge > par.tol_ge && iter_ge <= par.maxiter_ge

% prices given aggregates
w = (1-par.palpha)*par.A*(par.pa/L)^par.palpha;

% workers
for iz = 1:par.pnumgridz
    z = vgridz(iz);
    n = intensive_labor_supply(par.pbeta, par.pfrisch, par.peta, par.ptau, par.pa, z, w, par.r, T);
    c = ((1-par.ptau)*z*w*n + par.pa*R + T)/(1+par.pbeta);
    c(c<1e-10) = 1e-10;
    vn_lab(iz) = n;
    vc_lab(iz) = c;
    vaprime_lab(iz) = par.pbeta*c;
end

% non-workers
c_nonlab = (par.pb + par.pa*R + T)/(1+par.pbeta);
c_nonlab(c_nonlab<=1e-10) = 1e-10;
aprime_nonlab = par.pbeta*c_nonlab;

% extensive margin
vW = fnW(par.pbeta, par.pfrisch, par.peta, vc_lab, vn_lab, vaprime_lab);
N = fnN(par.pbeta, c_nonlab, aprime_nonlab);
vwork = vW > N;

% aggregates
L_new = sum(vn_lab.*vwork.*vgridz.*vpiz);
T_new = par.ptau*(w*L_new + par.r*par.pa) - par.pb*sum((~vwork).*vpiz);

% update
err_ge = max(abs([L_new-L; T_new-T].^2));
L = par.wt1*L + (1-par.wt1)*L_new;
T = par.wt2*T + (1-par.wt2)*T_new;

iter_ge = iter_ge + 1;
end
flag = err_ge <= par.tol_ge;
end