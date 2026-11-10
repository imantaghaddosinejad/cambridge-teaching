function [i_l, i_u, wt_l, wt_u] = fnInterp1dVec(x, grid, extrap)
% Vectorized linear interpolation
if nargin < 3
    extrap = false; 
end

    % Save original dimensions
    xsize = size(x);
    xvec = x(:);
    grid = grid(:);
    Ng = numel(grid);

    % interpolate 
    il_vec = sum(grid' < xvec,2);
    il_vec(il_vec<=1) = 1;
    il_vec(il_vec>=Ng) = Ng-1;
    iu_vec = il_vec+1;

    if extrap 
        % interpolation weights (linear + extrapolation)
        wtl_vec = (grid(iu_vec) - xvec) ./ (grid(iu_vec) - grid(il_vec)); 
        wtu_vec = 1 - wtl_vec;
    else
        % interpolation weights (linear + no extrapolation)
        wtl_vec = (grid(iu_vec) - xvec) ./ (grid(iu_vec) - grid(il_vec));
        wtl_vec(wtl_vec<0) = 0;
        wtl_vec(wtl_vec>1) = 1;    
        wtu_vec = 1 - wtl_vec;
    end

    % Restore original dimensions
    i_l  = reshape(il_vec,  xsize);
    i_u  = reshape(iu_vec,  xsize);
    wt_l = reshape(wtl_vec, xsize);
    wt_u = reshape(wtu_vec, xsize);
end