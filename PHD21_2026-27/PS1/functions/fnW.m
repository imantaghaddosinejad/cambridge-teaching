function vW = fnW(pbeta, pfrisch, peta, c, n, aprime)
    vW = log(c) - peta*n.^(1+1/pfrisch)/(1+1/pfrisch) + pbeta*log(aprime);
end