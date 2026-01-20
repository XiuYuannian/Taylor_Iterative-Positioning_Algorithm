function Jrho = jacobian_rho(u, W)
u = u(:).';                  
M = size(W,1);
eps_dh = 1e-12;  
D = u - W;                    
dx = D(:,1); dy = D(:,2); dz = D(:,3);
q  = dx.^2 + dy.^2;     
dh = sqrt(q);             % horizontal distance
r  = sqrt(q + dz.^2);     % 3D distance
dh_safe = max(dh, eps_dh);
r2 = r.^2;
% ---- 雅可比：Grho1 = ∂rho/∂u^T （严格按图片公式）----
r1 = max(r(1), 1e-12);
D1 = D(1,:);                   % 1x3
Grho1 = zeros(M-1, 3);
for k = 2:M
    rk = max(r(k), 1e-12);
    Dk = D(k,:);               % 1x3
    num = (r1^2)*Dk - (rk^2)*D1;
    den = (r1^3) * rk;
    Grho1(k-1,:) = num / den;  % 1x3
end
Jrho = Grho1;
end
