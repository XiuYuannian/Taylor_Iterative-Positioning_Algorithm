%% ===================== RMSE vs Target shift distance (Hybrid, b1/b2) ====================
clc; clear; close all;

%% ================== 场景：站点与目标基准 ==================
u0 = [4, 3.8, 2] * 1e3;   % 目标基准位置 (m)
W0 = [ ...
     0.6,  1.4,  0.2;
    -1.2, -0.6,  0.15;
     1.4, -0.4, -0.2;
    -0.5,  0.8,  0.12;
     1.3, -0.4, -0.25;
    -0.8,  1.0, -0.15] * 1e3;
M = size(W0,1);

%% ================== 噪声设置（固定）==================
sigma_w    = 5;       % 站位不确定性 std (m)  -> Q2
sigma_tdoa = 3;       % TDOA 距离差 std (m)
sigma_rad  = 0.003;   % AOA std (rad)
sigma_groa = 0.03;    % GROA std

Q2    = (sigma_w^2) * eye(3*M);
Q_ATG = make_Q_ATG(M, sigma_tdoa, sigma_rad, sigma_groa);   % z=[theta;beta;d;rho]

%% ================== 目标平移设置 ==================
d_list = 1:20;
step = 200;                  % 每步 200m
dir = [1, 0, 0]; dir = dir / norm(dir);

%% ================== Monte Carlo & 迭代 ==================
MC = 200;
rng(42);
maxIter = 60;
tol = 1e-6;

% 初值（可改成 warm-start：用上一点的估计作为下一点初值）
u_init = mean(W0,1) + [2000, -1500, 800];
w_init = reshape(W0.', [], 1);

%% ================== 存储：RMSE 与 CRB(1D) ==================
RMSE_b1_u = zeros(numel(d_list),1);
RMSE_b2_u = zeros(numel(d_list),1);
RMSE_b2_w = zeros(numel(d_list),1);

crb1d_u_nom   = zeros(numel(d_list),1);   % 在 W0 上算的 Hybrid CRB_u
crb1d_w_nom   = zeros(numel(d_list),1);   % 在 W0 上算的 Hybrid CRB_w
crb1d_w_avg   = zeros(numel(d_list),1);   % MC 平均后的 Hybrid CRB_w（推荐对比）
crb1d_u_avg   = zeros(numel(d_list),1);   % （可选）MC 平均后的 Hybrid CRB_u

%% ================== 主循环：目标位置变化 ==================
for k = 1:numel(d_list)
    u_true = u0 + dir * (d_list(k) * step);

    % ---- CRB（名义几何 W0）----
    [crb1d_u_nom(k), crb1d_w_nom(k)] = crb_1d_taylorb(u_true, W0, Q_ATG, Q2, "ATG");

    % ---- Monte Carlo ----
    e_b1_u = zeros(MC,1);
    e_b2_u = zeros(MC,1);
    e_b2_w = zeros(MC,1);

    crb_u_mc = zeros(MC,1);
    crb_w_mc = zeros(MC,1);

    for mc = 1:MC
        % 1) 每次 MC 随机真实站位（与 Q2 一致）
        ew = sigma_w * randn(M,3);
        W_true = W0 + ew;

        % 2) 真实观测（由 W_true 产生）+ 测量噪声 Q1
        [theta_t, beta_t, d_t, rho_t] = obs(u_true, W_true);
        z_true = [theta_t; beta_t; d_t; rho_t];
        z = z_true + chol(Q_ATG,'lower') * randn(length(z_true),1);

        % 3) b1：估计端用名义站位 W0（与 Q2 的“不确定性先验”语义一致）
        [u_hat1, ~] = taylor_b1(u_init, z, W0, Q_ATG, Q2, "ATG", maxIter, tol);
        e_b1_u(mc) = norm(u_hat1 - u_true);

        % 4) b2：v 用名义站位 W0；同时联合估计 u,w
        [u_hat2, w_hat2, ~] = taylor_b2(u_init, z, w_init, W0, Q_ATG, Q2, "ATG", maxIter, tol);
        e_b2_u(mc) = norm(u_hat2 - u_true);

        w_true_vec = reshape(W_true.', [], 1);
        e_b2_w(mc) = norm(w_hat2 - w_true_vec);   % sqrt(E[||e||^2]) 口径

        % 5) （关键）同一统计对象下的 CRB：在 W_true 上算 hybrid CRB 再平均
        [crb_u_mc(mc), crb_w_mc(mc)] = crb_1d_taylorb(u_true, W_true, Q_ATG, Q2, "ATG");
    end

    RMSE_b1_u(k) = sqrt(mean(e_b1_u.^2));
    RMSE_b2_u(k) = sqrt(mean(e_b2_u.^2));
    RMSE_b2_w(k) = sqrt(mean(e_b2_w.^2));

    crb1d_u_avg(k) = mean(crb_u_mc);
    crb1d_w_avg(k) = mean(crb_w_mc);

    fprintf('shift=%4dm | RMSE_b2_w=%.3f | CRB_w_nom=%.3f | CRB_w_avg=%.3f\n', ...
        d_list(k)*step, RMSE_b2_w(k), crb1d_w_nom(k), crb1d_w_avg(k));
end

%% ================== 绘图：目标 RMSE vs CRB ==================
x = d_list * step;

figure; hold on; grid on; box on;
plot(x, RMSE_b1_u, 'o', 'LineWidth', 1.6);
plot(x, RMSE_b2_u, '*', 'LineWidth', 1.6);
plot(x, crb1d_u_nom, '--o', 'LineWidth', 1.6);
plot(x, crb1d_u_avg, '--*', 'LineWidth', 1.6);
xlabel('Target shift distance (m)');
ylabel('Target position RMSE / bound (m)');
title('Target RMSE & Hybrid CRB vs Target Shift (ATG)');
legend('RMSE b1 (u)','RMSE b2 (u)','Hybrid CRB_u (W0)','Hybrid CRB_u (avg over W\_true)', ...
       'Location','northwest');
set(gca,'FontSize',12);

%% ================== 绘图：站位 RMSE vs CRB ==================
figure; hold on; grid on; box on;
q2_line = sqrt(trace(Q2)) * ones(size(x));   
plot(x, RMSE_b2_w,  'o',  'LineWidth', 1.6);
plot(x, crb1d_w_nom,'--o','LineWidth', 1.6);
% plot(x, crb1d_w_avg,'--*','LineWidth', 1.6);
plot(x, q2_line,    '-.', 'LineWidth', 1.6);   % <<< 新增

xlabel('Target shift distance (m)');
ylabel('Station RMSE / bound (m)');
title('Station RMSE & Hybrid CRB vs Target Shift (ATG)');
legend('RMSE b2 (w)', 'Hybrid CRB_w (W0)','\surd(tr(Q_2))', ...
       'Location','northwest');
set(gca,'FontSize',12);

%% ====================== local function ======================
function Q = make_Q_ATG(M, sigma_diffD, sigma_rad, sigma_groa)
nd = M-1;
n = M + M + nd + nd;                 % theta(M) + beta(M) + d(nd) + rho(nd)
Q = zeros(n,n);
Q(1:M,1:M) = (sigma_rad^2)*eye(M);                       % theta
Q(M+(1:M),M+(1:M)) = (sigma_rad^2)*eye(M);               % beta
Q(2*M+(1:nd),2*M+(1:nd)) = (sigma_diffD^2)*eye(nd);      % d
Q(2*M+nd+(1:nd),2*M+nd+(1:nd)) = (sigma_groa^2)*eye(nd); % rho
end