function [u_hat, info] = taylor_b1(u0, z, V, Q1, Q2, mode, maxIter, tol)
%TAYLOR_B1  Taylor-b1 迭代（对应式(4.45)）
%   u_{k+1} = u_k + (G1' * Qtil^{-1} * G1)^{-1} * G1' * Qtil^{-1} * (z - g(u_k,V))
%   Qtil(u) = Q1 + G2(u,V) * Q2 * G2(u,V)'
% Inputs
%   u0      : 1x3 or 3x1  初始目标位置
%   z       : Nz x 1      观测向量（顺序必须与 g_stack/G_u/G_w一致）
%   V       : M x 3       站位观测/名义站位（公式中的 v）
%   Q1      : Nz x Nz     观测噪声协方差（n 的协方差）
%   Q2      : 3M x 3M     站位误差协方差（m 的协方差）
%   maxIter : 最大迭代次数
%   tol     : 收敛阈值（||delta||）
% Outputs
%   u_hat   : 1x3         估计目标位置
%   info    : struct      迭代信息（可选）


u_hat = u0(:).';
Nz = length(z);
info.delta_norm = zeros(maxIter,1);
info.res_norm   = zeros(maxIter,1);
for it = 1:maxIter
    % ---------- 计算残差&雅可比  ----------
    [Gtheta2, Gbeta2, Gd2, Grho2] = jacobian_w(u_hat , V);
    [theta, beta, d, rho] = obs(u_hat , V);
    switch mode
        case "ATG"
            gk = [theta; beta;d; rho];
            G1 = [ jacobian_theta(u_hat , V);
                jacobian_beta(u_hat , V);
                jacobian_dist(u_hat , V);
                jacobian_rho(u_hat , V)
                ];
            G2 = [ Gtheta2;
                Gbeta2;
                Gd2;
                Grho2
                ];
        case "AT"
            gk = [theta; beta;d];
            G1 = [ jacobian_theta(u_hat , V);
                jacobian_beta(u_hat , V);
                jacobian_dist(u_hat , V)
                ];
            G2 = [ Gtheta2;
                Gbeta2;
                Gd2
                ];
        case "AG"
            gk = [theta; beta;rho];
            G1 = [ jacobian_theta(u_hat , V);
                jacobian_beta(u_hat , V);
                jacobian_rho(u_hat , V)
                ];
            G2 = [ Gtheta2;
                Gbeta2;
                Grho2
                ];
        case "TG"
            gk = [d; rho];
            G1 = [ jacobian_dist(u_hat , V);
                jacobian_rho(u_hat , V)
                ];
            G2 = [ Gd2;
                Grho2
                ];
    end
    % ---------- 3) 等效协方差 Qtil ----------
    r  = z - gk;
    Qtil = Q1 + G2 * Q2 * G2.'; % NzxNz
    % ---------- 4) 线性求解（避免显式求逆） ----------
    % y = Qtil^{-1} r,  Z = Qtil^{-1} G1
    y = Qtil \ r;
    Z = Qtil \ G1;
    % A = G1' Qtil^{-1} G1,  b = G1' Qtil^{-1} r
    A = G1.' * Z;     % 3x3
    b = G1.' * y;     % 3x1
    % ---------- 5) 更新 ----------
    delta = A \ b;    % 3x1
    u_new = u_hat + delta.';
    info.delta_norm(it) = norm(delta);
    info.res_norm(it)   = norm(r);
    u_hat = u_new;
    if norm(delta) < tol
        info.delta_norm = info.delta_norm(1:it);
        info.res_norm   = info.res_norm(1:it);
        info.iters = it;
        return;
    end
end

info.iters = maxIter;
end