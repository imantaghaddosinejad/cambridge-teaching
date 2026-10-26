%% Capital Demand and Supply
% Produce figures IIa and IIb in Aiyagari (1994) using fnSavingsDemand.m
% 
%%
%===============
% Figures IIa 
%===============
% savings demand Ea(r) vs capital demand K(r)
% pack parameters
par.pbeta = pbeta;
par.palpha = palpha;
par.pdelta = pdelta;
par.pA = pA;
par.pnumgrida = pnumgrida;
par.pnumgridz = pnumgridz;
par.pmina = pmina;
par.mtransz = mtransz;
par.vgridz = vgridz;
par.vgrida = vgrida;
par.tolpol = 1e-8;
par.toldist = toldist;
par.maxiter = 20000;

% aggregate savings 
vr = [linspace(-0.04, 0.035, 35), linspace(0.0352, 0.0412, 40)];
vr = unique([vr, r]); % include ge r so the curve passes through the point
vEa = zeros(size(vr));
vflag = zeros(numel(vr),2);
par.wt2 = 0.5;
par.wt3 = 0.5;
for ir = 1:numel(vr)
    [vEa(ir), vflag(ir,:)] = fnSavingsDemand(vr(ir), par);
end

% capital demand on its own dense grid (analytical)
vr2 = linspace(-pdelta+0.01, 0.08, 500); % stay off the -delta asymptote
vKd2 = supplyL*((vr2+pdelta)/(palpha*pA)).^(1/(palpha-1));

% plot
figure;
h1 = plot(vEa, vr, 'LineWidth', 2); % savings supply Ea(r)
hold on;
h2 = plot(vKd2, vr2, 'LineWidth', 2); % capital demand K(r)
yline(1/pbeta-1, '--', '\lambda = 1/\beta - 1', 'fontsize', 14);
yline(-pdelta, '--', '-\delta', 'fontsize', 14);
yline(0, '-k');
h3 = plot(K, r, 'ko', 'MarkerFaceColor', 'k'); % ge from main loop
hold off;
xlim([0, 3*K]);
ylim([-pdelta*1.1, 0.08]);
xlabel('Assets, Capital');
ylabel('r');
title('Steady state');
legend([h1, h2, h3], {'Ea(r)', 'K(r)', 'equilibrium'}, 'Location', 'southwest', 'FontSize', 10);
exportgraphics(gcf, '../figures/fig_capdemand_supply.png', 'Resolution', 300);


%===============
% figure IIa 
%===============
% Ea_w(r) for different borrowing limits (w fixed at ge)
vb = [0, 2]; % borrowing limits, a >= -b
vr3 = [linspace(-0.5, -0.065, 10), linspace(-0.06, 0.03, 40), linspace(0.0305, 0.0412, 30)];
vr3 = unique([vr3, r]); % include ge r so b = 0 curve passes through the point
pnatlimit = w*min(vgridz)/max(vr3); % natural limit at highest r

% guard against violating natural borrowing limit
if max(vb) >= pnatlimit
    error('b exceeds natural borrowing limit %.2f', pnatlimit);
end

% savings demand
par2 = par;
par2.w = w; % fix wage at ge
mEa = zeros(numel(vb), numel(vr3));
mflag1 = zeros(numel(vb), numel(vr3)); % pfi converged
mflag2 = zeros(numel(vb), numel(vr3)); % distribution converged
for ib = 1:numel(vb)
    par2.pmina = -vb(ib);
    par2.vgrida = par2.pmina + (pmaxa - par2.pmina)*y; % same curvature, starts at -b
    for ir = 1:numel(vr3)
        [mEa(ib,ir), vflagtemp] = fnSavingsDemand(vr3(ir), par2);
        mflag1(ib,ir) = vflagtemp(1);
        mflag2(ib,ir) = vflagtemp(2);
    end
end

% plot (linear, asset levels)
figure;
hold on;
vh = gobjects(numel(vb)+1, 1);
for ib = 1:numel(vb)
    % keep points where both pfi and distribution have converged, and r is below the steady-state interest rate
    vkeep = (mflag1(ib,:)==1) & (mflag2(ib,:)==1) & (vr3 < 1/pbeta-1);
    vh(ib) = plot(mEa(ib,vkeep), vr3(vkeep), 'LineWidth', 2);
    xline(-vb(ib), ':', sprintf('b = %g', vb(ib)), 'LabelVerticalAlignment', 'bottom'); % borrowing limit
end
vh(end) = plot(K, r, 'ko', 'MarkerFaceColor', 'k'); % ge (b = 0, w = w*)
yline(1/pbeta-1, '--', '\lambda = 1/\beta - 1', 'fontsize', 14);
yline(0, '-k');
xline(0, '-k');
hold off;
xlim([-max(vb)-0.5, 1.75*K]);
ylim([-0.01, 0.06]);
xlabel('Assets');
ylabel('r');
title('Interest rate vs per capita assets (w fixed)');
legend(vh, [compose('b = %g', vb), {'equilibrium'}], 'Location', 'east', 'FontSize', 10);
exportgraphics(gcf, '../figures/fig_borrowing_limits.png', 'Resolution', 300);