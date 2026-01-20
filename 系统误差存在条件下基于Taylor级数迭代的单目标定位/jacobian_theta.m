function Jtheta = jacobian_theta(u, W)
u = u(:).';                  
M = size(W,1);
D = u - W;                    
dx = D(:,1); dy = D(:,2); dz = D(:,3);
q  = dx.^2 + dy.^2;
dh = sqrt(q);             % horizontal distance
r  = sqrt(q + dz.^2);     % 3D distance
% (a) theta
Jtheta = [ -dy./q,  dx./q,  zeros(M,1) ];
end



