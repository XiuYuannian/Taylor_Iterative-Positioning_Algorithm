function [Gtheta2, Gbeta2, Gd2, Grho2] = jacobian_w(u, W)
% 观测函数 g(u,w)=[theta; beta; d; rho] 对 w 的雅可比
% u: 1x3 目标位置
% W: Mx3 站位置 (w_m)
% 输出：
%   Gtheta2: M x 3M
%   Gbeta2 : M x 3M
%   Gd2    : (M-1) x 3M
%   Grho2  : (M-1) x 3M

u = u(:).';
M = size(W,1);

D = u - W;                 % Mx3, D(m,:) = u - w_m = [dx dy dz]
dx = D(:,1); dy = D(:,2); dz = D(:,3);

q  = dx.^2 + dy.^2;
dh = sqrt(q);
r  = sqrt(q + dz.^2);

epsv = 1e-12;
dh_safe = max(dh, epsv);
r_safe  = max(r,  epsv);
q_safe  = max(q,  epsv);

Gtheta2 = zeros(M, 3*M);
Gbeta2  = zeros(M, 3*M);
Gd2     = zeros(M-1, 3*M);
Grho2   = zeros(M-1, 3*M);

%% ---------- 1) theta 对 w：每行只作用在对应 w_m 的 3 列 ----------
% ∂theta_m/∂w_m^T = [ (y_t - y_m)/||I3||^2 , -(x_t-x_m)/||I3||^2 , 0 ]
for m = 1:M
    cols = (3*(m-1)+1):(3*m);
    Gtheta2(m, cols) = [ dy(m)/q_safe(m), -dx(m)/q_safe(m), 0 ];
end

%% ---------- 2) beta 对 w：每行只作用在对应 w_m 的 3 列 ----------
% ∂beta_m/∂w_m^T = [ (dx*dz)/(dh*r^2), (dy*dz)/(dh*r^2), -dh/(r^2) ]
r2 = r_safe.^2;
for m = 1:M
    cols = (3*(m-1)+1):(3*m);
    Gbeta2(m, cols) = [ (dx(m)*dz(m))/(dh_safe(m)*r2(m)), ...
                        (dy(m)*dz(m))/(dh_safe(m)*r2(m)), ...
                       -dh_safe(m)/r2(m) ];
end

%% ---------- 3) d 对 w：d_n = r_n - r_1, n=2..M ----------
% 行 idx = n-1 对应站 n
% ∂d/∂w1^T = (u-w1)^T/||u-w1||
% ∂d/∂wn^T = (wn-u)^T/||u-wn||
r1 = r_safe(1);
u_minus_w1 = D(1,:);        % (u - w1)
for n = 2:M
    row = n-1;

    % w1 block
    cols1 = 1:3;
    Gd2(row, cols1) = (u_minus_w1 / r1);

    % wn block
    colsn = (3*(n-1)+1):(3*n);
    wn_minus_u = -D(n,:);   % (w_n - u)
    Gd2(row, colsn) = (wn_minus_u / r_safe(n));
end

%% ---------- 4) rho 对 w：rho_n = r_n / r_1, n=2..M ----------
% ∂rho/∂w1^T = ||u-wn|| * (u-w1)^T / ||u-w1||^3
% ∂rho/∂wn^T = (wn-u)^T / (||u-w1||*||u-wn||)
r1_3 = r1^3;
for n = 2:M
    row = n-1;

    rn = r_safe(n);
    cols1 = 1:3;
    Grho2(row, cols1) = (rn * u_minus_w1) / r1_3;     % = rn*(u-w1)/r1^3

    colsn = (3*(n-1)+1):(3*n);
    wn_minus_u = -D(n,:);                              % (w_n - u)
    Grho2(row, colsn) = wn_minus_u / (r1 * rn);
end

end
