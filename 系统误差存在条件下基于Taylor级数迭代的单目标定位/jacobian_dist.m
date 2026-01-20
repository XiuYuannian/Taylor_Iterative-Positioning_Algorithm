function Jdist = jacobian_dist(u, W)
u = u(:).';                  
M = size(W,1);
eps_dh = 1e-12;  
D = u - W;                    
dx = D(:,1); dy = D(:,2); dz = D(:,3);
q  = dx.^2 + dy.^2;
dh = sqrt(q);             % horizontal distance
r  = sqrt(q + dz.^2);     % 3D distance
R = sqrt(sum(D.^2, 2));     % Mx1, r_m = ||u-w_m||
% 雅可比：每行 = (u-w_k)/||u-w_k|| - (u-w_1)/||u-w_1||

unit = D ./ max(R, eps_dh);  % Mx3, unit(m,:) = (u-w_m)/||u-w_m||
Jdist = unit(2:end,:) - unit(1,:);
end
