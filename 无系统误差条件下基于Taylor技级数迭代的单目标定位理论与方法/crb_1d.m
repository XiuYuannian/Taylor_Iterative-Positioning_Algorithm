%% ================== CRB(1D) ==================
function crb1d = crb_1d(u_true, W, Q, mode)
Qi = diag(1./diag(Q));
switch mode
    case "ATG"
        G = [ jacobian_theta(u_true, W);
              jacobian_beta(u_true, W); 
              jacobian_dist(u_true, W);
              jacobian_rho(u_true, W)
              ];
    case "AT"
        G = [ jacobian_theta(u_true, W);
              jacobian_beta(u_true, W);
              jacobian_dist(u_true, W)
             ];
    case "AG"
        G = [ jacobian_theta(u_true, W);
              jacobian_beta(u_true, W);
              jacobian_rho(u_true, W)
             ];
    case "TG"
        G = [ jacobian_dist(u_true, W);
            jacobian_rho(u_true, W)
            ];
end

CRB = inv(G' * Qi * G);
crb1d = sqrt(trace(CRB));  % 三维 -> 一维
end