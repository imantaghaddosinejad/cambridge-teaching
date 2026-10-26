function vM = fnMom(w, vn_lab, vwork, vgridz, vpiz, pb)
% targeted moments
vmassW = vwork.*vpiz;
emp = sum(vmassW);
vinc = w*vgridz.*vn_lab;
meaninc = sum(vmassW.*vinc)/emp;
sdinc = sqrt(sum(vmassW.*(vinc-meaninc).^2)/emp);
vM = [sum(vmassW.*vn_lab)/emp, ...    % avg hours of workers
    1-emp, ...                         % unemployment rate
    pb/meaninc, ...                    % b / avg wage income of workers
    sdinc/meaninc];                    % std/mean wage income of workers
end