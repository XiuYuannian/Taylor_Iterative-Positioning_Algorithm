%% ================== 组合：观测与雅可比装配 + WLS求解 ==================
function u_hat = solve_wls(u0, z, W, Q, mode, maxIter, tol)
u_hat = u0;
Qi = diag(1./diag(Q));   % Q为对角阵
for it = 1:maxIter
    [theta, beta, d, rho] = obs(u_hat, W);
    switch mode
        case "ATG"
            g = [theta; beta;d; rho];
            G = [ jacobian_theta(u_hat, W);
                  jacobian_beta(u_hat, W);
                  jacobian_dist(u_hat, W);
                  jacobian_rho(u_hat, W)
                   ];
        case "AT"
            g = [theta; beta; d];
            G = [ jacobian_theta(u_hat, W);
              jacobian_beta(u_hat, W);
              jacobian_dist(u_hat, W)
             ];
        case "AG"
            g = [theta; beta;rho];
            G = [ jacobian_theta(u_hat, W);
              jacobian_beta(u_hat, W);
              jacobian_rho(u_hat, W)
             ];
         case "TG"
            g = [d;rho];
            G = [ jacobian_dist(u_hat, W);
              jacobian_rho(u_hat, W)
             ];
        otherwise
            error("Unknown mode");
    end

    r = z - g;
    %有时求解会因为参数设置导致无法收敛，可尝试更改delta的求解方式：加 LM 阻尼
    delta = (G' * Qi * G) \ (G' * Qi * r);
%     lambda = 1e-4;   
%     A = G' * Qi * G;
%     b = G' * Qi * r;
%     delta = (A + lambda*eye(size(A))) \ b;
    u_new = u_hat + delta';
    if norm(delta) < tol
        u_hat = u_new;
        break;
    end
    u_hat = u_new;
end
end
