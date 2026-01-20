function Gw = G_w(u, W)
[Gtheta2, Gbeta2, Gd2, Grho2] = jacobian_w(u, W);
Gw = [Gtheta2; Gbeta2; Gd2; Grho2];
end

