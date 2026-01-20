function Jbeta = jacobian_beta(u, W)
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
% (b) beta
Jbeta = [ -(dx.*dz)./(dh_safe.*r2), ...
          -(dy.*dz)./(dh_safe.*r2), ...
           dh_safe./r2 ];
end
