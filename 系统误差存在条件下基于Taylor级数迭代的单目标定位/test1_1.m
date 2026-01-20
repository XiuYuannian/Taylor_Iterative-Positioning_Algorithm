%% ====================目标位置估计均方根误差随观测量扰动参数的变化曲线（Taylor—a）=====================
clc; clear all; close all;
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

%% ================== 误差设置 ==================
sigma_w = 5;                       % 观测站位置误差 std (m)
deta1 = 1:20;                      % 扰动参数
sigma_aoa_list  = 0.001 * deta1;   % rad
sigma_tdoa_list = 0.3   * deta1;   % m (这里是“距离差”的std)
sigma_groa_list = 0.001 * deta1;   % 无量纲
%% ================== Monte Carlo & 迭代参数 ==================
MC = 200;
rng(42);
maxIter = 60;
tol = 1e-6;
% 初值（你也可以改成更贴近真实）
% u_init = mean(W0,1) + [2000, -1500, 800];
u_init = 100 + [2000, -1500, 800];
w_init = reshape(W0.', [], 1);
ew = sigma_w * randn(M,3);     % 
W_obs = W0 + ew;
Q2 = (sigma_w^2) * eye(3*M);
%% ================== 存储：MSE（m^2） ==================
RMSE1_pos = zeros(numel(deta1),1);
RMSE2_pos = zeros(numel(deta1),1);
RMSE3_pos = zeros(numel(deta1),1);
crb1d_u = zeros(numel(deta1),1);  
crb1d_w = zeros(numel(deta1),1);   
crb1d_a = zeros(numel(deta1),1); 
Cov_1d = zeros(numel(deta1),1);   
%% ================== 主循环：随 deta1 变化 ==================
for k = 1:numel(deta1)
    sig_aoa  = sigma_aoa_list(k);
    sig_tdoa = sigma_tdoa_list(k);
    sig_groa = sigma_groa_list(k);
    Q1 = make_R(M, sig_tdoa, sig_groa, sig_aoa);
    [crb1d_u(k), crb1d_w(k)] = crb_1d_taylorb(u_true0, W0, Q1, Q2, "ATG");
    [crb1d_a(k), CRB_a]      = crb_1d(u_true0, W0, Q1, "ATG");
    Cov_1d(k)                = cov_taylor_a(CRB_a, u_true0, W0, Q1, Q2, "ATG");
    err1 = zeros(MC,1);
    for mc = 1:MC
        % --- 每次MC随机真实站位（与Q2一致） ---
        ew     = sigma_w * randn(M,3);
        W_true = W0 + ew;
        % --- 观测由真实站位产生 ---
        z_true = g_stack(u_true0, W_true);
        z      = z_true + chol(Q1,'lower')*randn(length(z_true),1);
        % --- 估计使用名义站位（你“掌握”的站位） ---
        u_hat1 = taylor_a(u_init, z, W0, Q1, "ATG", maxIter, tol);
        err1(mc) = norm(u_hat1 - u_true0);
    end
    RMSE1_pos(k) = sqrt(mean(err1.^2));
end

%% ================== 绘图：MSE 随 deta1 变化 ==================
figure; grid on; box on; hold on;

plot(deta1, RMSE1_pos, '*', 'LineWidth', 1.8, 'MarkerSize', 6);
plot(deta1, Cov_1d,   '--o', 'LineWidth', 1.8, 'MarkerSize', 6);
plot(deta1, crb1d_a,  '--d','LineWidth', 1.8, 'MarkerSize', 6);
plot(deta1, crb1d_u,  '--^','LineWidth', 1.8, 'MarkerSize', 6);

xlabel('\delta_1','FontSize',12);
ylabel('Position Error (m)','FontSize',12);
 
title('RMSE and CRB versus \delta_1 (with station position uncertainty)', ...
      'FontSize',13);

legend({ ...
    'Taylor-a RMSE', ...
    'Taylor-a CRB', ...
    'Hybrid CRB_a',...
    'Hybrid CRB_b'}, ...
    'Location','northwest','FontSize',11);

set(gca,'FontSize',12);
