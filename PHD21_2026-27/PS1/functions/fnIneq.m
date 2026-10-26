function [gini, varlog] = fnIneq(vy, vmass)
% gini and variance of log income for a discrete weighted distribution
keep = vmass > 0;
vy = vy(keep);
vmass = vmass(keep)/sum(vmass(keep));

% gini from the lorenz curve
[vy, idx] = sort(vy);
vmass = vmass(idx);
vlorenz = cumsum(vmass.*vy)/sum(vmass.*vy);
gini = 1 - sum(vmass.*([0; vlorenz(1:end-1)] + vlorenz));

% variance of log income
meanlog = sum(vmass.*log(vy));
varlog = sum(vmass.*(log(vy) - meanlog).^2);
end
