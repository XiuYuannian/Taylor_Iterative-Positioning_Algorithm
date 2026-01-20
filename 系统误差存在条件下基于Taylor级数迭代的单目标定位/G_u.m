function Gu = G_u(u, W)
M = size(W,1);

Jd   = jacobian_dist(u, W);    % (M-1)x3
Jrho = jacobian_rho(u, W);     % (M-1)x3
Jth  = jacobian_theta(u, W);   % Mx3
Jbe  = jacobian_beta(u, W);    % Mx3

Gu = [Jth; Jbe; Jd; Jrho];
end