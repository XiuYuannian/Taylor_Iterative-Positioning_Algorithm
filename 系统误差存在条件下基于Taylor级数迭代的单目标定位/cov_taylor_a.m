function Cov_1d = cov_taylor_a(CRB_a, u_true, W, Q1, Q2,mode)
[Gtheta2, Gbeta2, Gd2, Grho2] = jacobian_w(u_true, W);
switch mode
    case "ATG"
        G1 = [ jacobian_theta(u_true, W);
              jacobian_beta(u_true, W); 
              jacobian_dist(u_true, W);
              jacobian_rho(u_true, W)
              ];
        G2 = [ Gtheta2;
              Gbeta2;
              Gd2;
              Grho2
              ];
    case "AT"
        G1 = [ jacobian_theta(u_true, W);
              jacobian_beta(u_true, W);
              jacobian_dist(u_true, W)
             ];
        G2 = [ Gtheta2;
              Gbeta2;
              Gd2
             ];
    case "AG"
        G1 = [ jacobian_theta(u_true, W);
              jacobian_beta(u_true, W);
              jacobian_rho(u_true, W)
             ];
        G2 = [ Gtheta2;
              Gbeta2;
              Grho2
             ];
end

% 计算 A = Q1^{-1} G1,  B = Q1^{-1} G2
A = Q1 \ G1;      % Nz x n
B = Q1 \ G2;      % Nz x M2

% 计算中间量：T = G1' Q1^{-1} G2 = G1' * (Q1\G2)
T = G1.' * B;     % n x M2

% 附加项：CRB_a * T * Q2 * T' * CRB_a
extraTerm = CRB_a * (T * Q2 * T.') * CRB_a;
% 总协方差
Cov_u = CRB_a + extraTerm;
% 数值对称化（避免浮点误差导致非对称）
% Cov_u = (Cov_u + Cov_u.')/2;
% extraTerm = (extraTerm + extraTerm.')/2;
Cov_1d = sqrt(trace(Cov_u));  % 三维 -> 一维

end