function [u_hat, w_hat, info] = taylor_b2(u0,z, w0,v, Q1, Q2, mode,maxIter, tol)
% 模型：
%   z = g(u,w) + n,   n ~ N(0, Q1)
%   v = w + m,        m ~ N(0, Q2)
% 迭代：
%   [u;w]_{k+1} = [u;w]_k + (J' W J)^{-1} J' W r
%   其中 J = [G1, G2; 0, I],  W = diag(Q1^{-1}, Q2^{-1})
%   r = [z - g(u,w); v - w]
% Inputs:
%   u0 : 1x3 or 3x1      初始目标
%   w0 : 3M x 1          初始站位向量（堆叠 [w1;...;wM]）
%   z  : Nz x 1          主观测
%   v  : 3M x 1          站位观测
%   Q1 : Nz x Nz         主观测噪声协方差
%   Q2 : 3M x 3M         站位观测噪声协方差
%   maxIter, tol
% Outputs:
%   u_hat : 1x3
%   w_hat : 3M x 1
%   info  : struct (收敛信息)
u_hat = u0(:).';
w_hat = w0(:);
v_vec = reshape(v.', [], 1);   % 6x3 -> 18x1
info.delta_norm = zeros(maxIter,1);
info.res_norm   = zeros(maxIter,1);
% 为数值稳定起见用线性方程求解，不显式 inv
for it = 1:maxIter
    % -------- 1) 当前站位矩阵形式 Wk (Mx3) --------
    Wk = reshape(w_hat, 3, []).';   % Mx3
    % -------- 2) 预测观测 & 残差 --------
    gk = g_stack(u_hat, Wk);        % Nzx1
    r1 = z - gk;                    % Nzx1
    %r2 = v - w_hat;                 % 3Mx1
    r2    = v_vec - w_hat;         % 18x1
    % -------- 3) 雅可比 --------
    % G1 = ∂g/∂u^T : Nzx3
    % G2 = ∂g/∂w^T : Nzx3M
    [Gtheta2, Gbeta2, Gd2, Grho2] = jacobian_w(u_hat , Wk);
    switch mode
        case "ATG"
            G1 = [ jacobian_theta(u_hat , Wk);
                jacobian_beta(u_hat ,Wk);
                jacobian_dist(u_hat , Wk);
                jacobian_rho(u_hat ,Wk)
                ];
            G2 = [ Gtheta2;
                Gbeta2;
                Gd2;
                Grho2
                ];
        case "AT"
            G1 = [ jacobian_theta(u_hat , Wk);
                jacobian_beta(u_hat , Wk);
                jacobian_dist(u_hat , Wk)
                ];
            G2 = [ Gtheta2;
                Gbeta2;
                Gd2
                ];
        case "AG"
            G1 = [ jacobian_theta(u_hat , Wk);
                jacobian_beta(u_hat , Wk);
                jacobian_rho(u_hat , Wk)
                ];
            G2 = [ Gtheta2;
                Gbeta2;
                Grho2
                ];
        case "TG"
            G1 = [ jacobian_dist(u_hat , Wk);
                jacobian_rho(u_hat , Wk)
                ];
            G2 = [ Gd2;
                Grho2
                ];
    end
    % -------- 4) 组装正规方程 (J'WJ) 和 右端 (J'Wr) --------
    % 记 Q1^{-1}*X = Q1\X, Q2^{-1}*X = Q2\X
    % A = G1' Q1^{-1} G1
    A = G1.' * (Q1 \ G1);
    % B = G1' Q1^{-1} G2
    B = G1.' * (Q1 \ G2);
    % C = G2' Q1^{-1} G1
    C = G2.' * (Q1 \ G1);
    % D = G2' Q1^{-1} G2 + Q2^{-1}
    D = G2.' * (Q1 \ G2) + (Q2 \ eye(size(Q2)));
    % 右端项：
    bu = G1.' * (Q1 \ r1);
    bw = G2.' * (Q1 \ r1) + (Q2 \ r2);
    % 联合增量 [du; dw]
    % [A B; C D] [du;dw] = [bu;bw]
    H = [A, B;
         C, D];
    b = [bu;
         bw];
    delta = H \ b;
    du = delta(1:3);
    dw = delta(4:end);
    % -------- 5) 更新 --------
    u_new = u_hat + du.';
    w_new = w_hat + dw;
    info.delta_norm(it) = norm(delta);
    info.res_norm(it)   = norm([r1; r2]);
    u_hat = u_new;
    w_hat = w_new;
    if norm(delta) < tol
        info.delta_norm = info.delta_norm(1:it);
        info.res_norm   = info.res_norm(1:it);
        info.iters = it;
        return;
    end
end

info.iters = maxIter;
end