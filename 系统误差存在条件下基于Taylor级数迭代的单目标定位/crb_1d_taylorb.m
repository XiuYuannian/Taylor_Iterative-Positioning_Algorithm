%% ================== CRB(1D) ==================
function [crb1d_u,crb1d_w] = crb_1d_taylorb(u_true, W, Q1,Q2, mode)
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
    case "TG"
        G1 = [ jacobian_dist(u_true, W);
            jacobian_rho(u_true, W)
            ];
        G2 = [ Gd2;
            Grho2
            ];
end
A = G1' * (Q1 \ G1);
B = G1' * (Q1 \ G2);
C = G2' * (Q1 \ G1);
D = G2' * (Q1 \ G2) + (Q2 \ eye(size(Q2)));   % = G2'Q1^{-1}G2 + Q2^{-1}
F = [A, B;
     C, D];
CRB_uw = inv(F);
CRB_u = inv(A - B*(D\C));         
CRB_w = inv(D - C*(A\B));           
crb1d_u = sqrt(trace(CRB_u));  % 三维 -> 一维
crb1d_w = sqrt(trace(CRB_w));  % 三维 -> 一维
end

