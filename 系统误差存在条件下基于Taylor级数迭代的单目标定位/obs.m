function [theta, beta, d, rho] = obs(u, W)
% u : 1x3 目标位置
% W : Mx3 观测站位置 w_m
% theta(m) : 方位角(全M个)
% beta(m)  : 仰角(全M个)
% d(n)     : 距离差观测 r_n - r_1, n=2..M   -> (M-1)x1
% rho(n)   : 增益比观测 r_n / r_1, n=2..M   -> (M-1)x1
% Gd1      : ∂d/∂u^T   -> (M-1)x3  (对应图片 G_{d1})
% Grho1    : ∂rho/∂u^T -> (M-1)x3  (对应图片 G_{rho1})
u = u(:).';                  
M = size(W,1);
D = u - W;                    
dx = D(:,1); dy = D(:,2); dz = D(:,3);
q  = dx.^2 + dy.^2;
dh = sqrt(q);
r  = sqrt(q + dz.^2);        
theta = atan2(dy, dx);
beta  = atan2(dz, max(dh, 1e-12));
d   = r(2:end) - r(1);       
rho = r(2:end) ./ r(1);
end