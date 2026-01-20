%% ====================目标位置估计均方根误差随系统参量扰动参数的变化曲线（Taylor—a）==========
clc; clear; close all;
%% ================== 场景：站点与目标（名义值） ==================
u_true0 = [4, 3.8, 2] * 1e3;   % 目标真值 (m)
W0 = [ ...
     0.6,  1.4,  0.2;
    -1.2, -0.6,  0.15;
     1.4, -0.4, -0.2;
    -0.5,  0.8,  0.12;
     1.3, -0.4, -0.25;
    -0.8,  1.0, -0.15] * 1e3; % 名义站位 (m)
M = size(W0,1);

%% ================== 观测噪声设置 ==================
sigma_aoa  = 0.003;   % rad
sigma_tdoa = 3;       % (你的 make_R 里若用 ns，请保证单位一致)
sigma_groa = 0.03;    % (同上)

deta2 = 1:20;                     % 扰动参数
sigma_w_list = 0 + deta2;         % 站位误差标准差 (m) —— 

%% ================== Monte Carlo & 迭代参数 ==================
MC = 200;
rng(42);
maxIter = 60;
tol = 1e-6;

% 初值（可按需要调整）
u_init = mean(W0,1) + [2000, -1500, 800];
w_init = reshape(W0.', [], 1);
%% ===========观测量协方差矩阵（固定）======================
Q1 = make_R(M, sigma_tdoa, sigma_groa, sigma_aoa);

%% ================== 存储 ==================
RMSE1_pos = zeros(numel(deta2),1);
RMSE2_pos = zeros(numel(deta2),1);
RMSE3_pos = zeros(numel(deta2),1);
RMSE3_pos_w = zeros(numel(deta2),1);
crb1d_u   = zeros(numel(deta2),1);   % Hybrid CRB (含站参数先验) - 目标
crb1d_w   = zeros(numel(deta2),1);   % Hybrid CRB - 站参数
crb1d_a   = zeros(numel(deta2),1);   % 仅观测噪声下的 CRB_a（你 crb_1d 的定义）
Cov_1d    = zeros(numel(deta2),1);   % cov_taylor_a 的 1D 方差（或均方根）

%% ================== 主循环：随 sigma_w 变化 ==================
for k = 1:numel(deta2)
    sigma_w = sigma_w_list(k);           % (m)
    Q2 = (sigma_w^2) * eye(3*M);         % 站位先验协方差

    % -------- 理论量：对每个 k 只算一次（不要放在 MC 里）--------
    [crb1d_u(k), crb1d_w(k)] = crb_1d_taylorb(u_true0, W0, Q1, Q2, "ATG");
    [crb1d_a(k), CRB_a]      = crb_1d(u_true0, W0, Q1, "ATG");
    Cov_1d(k)                = cov_taylor_a(CRB_a, u_true0, W0, Q1, Q2, "ATG");

    % -------- Monte Carlo --------
    err1 = zeros(MC,1);
    err2 = zeros(MC,1);
    err3 = zeros(MC,1);
    err3_w = zeros(MC,1);

    for mc = 1:MC
        ew     = sigma_w * randn(M,3);
        W_true = W0 + ew;
        % --- 观测由真实站位产生 ---
        z_true = g_stack(u_true0, W_true);
        z      = z_true + chol(Q1,'lower')*randn(length(z_true),1);
        % --- taylor_a ---
        u_hat1 = taylor_a(u_init, z, W0, Q1, "ATG", maxIter, tol);
        err1(mc) = norm(u_hat1 - u_true0);
        % --- taylor_b1 (默认采用"ATG")---
        [u_hat2, ~] = taylor_b1(u_init, z, W0, Q1, Q2,"ATG", maxIter, tol);
        err2(mc) = norm(u_hat2 - u_true0);
        % --- taylor_b2 ---
        [u_hat3, w_hat3, ~] = taylor_b2(u_init,z, w_init,W0, Q1, Q2,"ATG", maxIter, tol);
        err3(mc) = norm(u_hat3 - u_true0);
        % 真实站位向量（3M×1）
        w_true_vec = reshape(W_true.', [], 1);
        % 站位估计误差（3M维）
        err3_w(mc) = norm(w_hat3 - w_true_vec);
    end
    RMSE1_pos(k) = sqrt(mean(err1.^2));
    RMSE2_pos(k) = sqrt(mean(err2.^2));
    RMSE3_pos(k) = sqrt(mean(err3.^2));
    RMSE3_pos_w(k) = sqrt(mean(err3_w.^2));
end

%% ================== 绘图 ==================
%% ================== 绘图：MSE 随 deta1 变化 ==================
figure; grid on; box on; hold on;

plot(deta2, RMSE2_pos, '*', 'LineWidth', 1.8, 'MarkerSize', 6);
plot(deta2, RMSE3_pos, 'd', 'LineWidth', 1.8, 'MarkerSize', 6);
plot(deta2, crb1d_a,  '--o','LineWidth', 1.8, 'MarkerSize', 6);
plot(deta2, crb1d_u,  '--^','LineWidth', 1.8, 'MarkerSize', 6);

xlabel('\delta_2','FontSize',12);
ylabel('Position Error (m)','FontSize',12);
 
title('RMSE and CRB versus \delta_1 (with station position uncertainty)', ...
      'FontSize',13);

legend({ ...
    'Taylor-b1 RMSE', ...
    'Taylor-b2 RMSE', ...
    'Hybrid CRB_a',...
    'Hybrid CRB_b'}, ...
    'Location','northwest','FontSize',11);

set(gca,'FontSize',12);

%% ================
figure; grid on; box on; hold on;
plot(deta2, RMSE3_pos_w, 'd', 'LineWidth', 1.8, 'MarkerSize', 6);
plot(deta2, crb1d_w,  '--^','LineWidth', 1.8, 'MarkerSize', 6);
xlabel('\delta_2','FontSize',12);
ylabel('Position Error (m)','FontSize',12);
title('RMSE and CRB versus \delta_1 (with station position uncertainty)', ...
      'FontSize',13);
legend({ ...
    'Taylor-b2(w) RMSE', ...
    'Hybrid CRB_b'}, ...
    'Location','northwest','FontSize',11);
set(gca,'FontSize',12);