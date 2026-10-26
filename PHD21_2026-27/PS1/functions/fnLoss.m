function err = fnLoss(x, par, vtarget, mweight)
% x = log([peta, pb, psiggz])
par.peta   = exp(x(1));
par.pb     = exp(x(2));
par.psiggz = exp(x(3));

% z grid for this psiggz (iid: every row of mtransz is the invariant distribution)
[mtransz, vlogz] = fnTauchen(par.prhoz, par.pmuz, par.psiggz^2, par.pnumgridz, 3.0);
vgridz = exp(vlogz);
vpiz = mtransz(1,:)';

[~, ~, w, vn_lab, ~, vwork, flag] = fnSolveGE(par, vgridz, vpiz, par.L0, par.T0);
if ~flag || sum(vwork.*vpiz) < 1e-6
    err = 1e10;     % ge failed or nobody works
    return;
end
vM = fnMom(w, vn_lab, vwork, vgridz, vpiz, par.pb);
verr = vM - vtarget;
err = verr*mweight*verr';
end
