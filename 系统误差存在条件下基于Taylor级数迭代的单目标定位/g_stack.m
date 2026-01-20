function z = g_stack(u, W)
[theta, beta, d, rho] = obs(u, W);
z = [theta; beta; d; rho];
end