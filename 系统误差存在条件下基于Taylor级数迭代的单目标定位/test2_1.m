%% ====================观测站位置估计均方根误差随系统参量扰动参数的变化曲线（Taylor—b2）=====================
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
RMSE3_pos_w = zeros(numel(deta1),1);
crb1d_u = zeros(numel(deta1),1);  
crb1d_w = zeros(numel(deta1),1);   
crb1d_a = zeros(numel(deta1),1); 
Cov_1d = zeros(numel(deta1),1);   
%% ================== 主循环：随 deta1 变化 ==================
for k = 1:numel(deta1)
    sig_aoa  = sigma_aoa_list(k);
    sig_tdoa = sigma_tdoa_list(k);
    sig_groa = sigma_groa_list(k);
    Q_ATG = make_Q_ATG(M, sig_tdoa, sig_aoa, sig_groa);
    Q_AT  = make_Q_AT(M,  sig_tdoa, sig_aoa);
    Q_AG  = make_Q_AG(M,  sig_aoa,   sig_groa);
    [crb1d_u(k), crb1d_w(k)] = crb_1d_taylorb(u_true0, W0, Q_ATG, Q2, "ATG");
    [crb1d_a(k), CRB_a]      = crb_1d(u_true0, W0, Q_ATG, "ATG");
    Cov_1d(k)                = cov_taylor_a(CRB_a, u_true0, W0,Q_ATG, Q2, "ATG");
    err1 = zeros(MC,1);
    err2 = zeros(MC,1);
    err3 = zeros(MC,1);
    err3_w = zeros(MC,1);
    for mc = 1:MC
        % --- 每次MC随机真实站位（与Q2一致） ---
        ew     = sigma_w * randn(M,3);
        W_true = W0 + ew;
        % --- 观测由真实站位产生 ---
        z_true = g_stack(u_true0, W_true);
        z      = z_true + chol(Q_ATG,'lower')*randn(length(z_true),1);
        % --- taylor_a ---
        u_hat1 = taylor_a(u_init, z, W0, Q_ATG, "ATG", maxIter, tol);
        err1(mc) = norm(u_hat1 - u_true0);  
        % --- taylor_b1 (默认采用"ATG")---
        [u_hat2, ~] = taylor_b1(u_init, z, W0, Q_ATG, Q2, "ATG",maxIter, tol);
        err2(mc) = norm(u_hat2 - u_true0);
        % --- taylor_b2 ---
        [u_hat3, w_hat3, ~] = taylor_b2(u_init,z, w_init, W0, Q_ATG, Q2, "ATG",maxIter, tol);
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

%% ================== 绘图：MSE 随 deta1 变化 ==================
figure; grid on; box on; hold on;

plot(deta1, RMSE2_pos, '*', 'LineWidth', 1.8, 'MarkerSize', 6);
plot(deta1, RMSE3_pos, 'd', 'LineWidth', 1.8, 'MarkerSize', 6);
plot(deta1, crb1d_a,  '--o','LineWidth', 1.8, 'MarkerSize', 6);
plot(deta1, crb1d_u,  '--^','LineWidth', 1.8, 'MarkerSize', 6);

xlabel('\delta_1','FontSize',12);
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

plot(deta1, RMSE3_pos_w, 'd', 'LineWidth', 1.8, 'MarkerSize', 6);
plot(deta1, crb1d_w,  '--^','LineWidth', 1.8, 'MarkerSize', 6);

xlabel('\delta_1','FontSize',12);
ylabel('Position Error (m)','FontSize',12);
 
title('RMSE and CRB versus \delta_1 (with station position uncertainty)', ...
      'FontSize',13);

legend({ ...
    'Taylor-b2(w) RMSE', ...
    'Hybrid CRB_b'}, ...
    'Location','northwest','FontSize',11);
set(gca,'FontSize',12);



%% ======================函数区=================================
function Q = make_Q_ATG(M,sigma_diffD, sigma_rad, sigma_groa)
nd = M-1;
n = nd +nd+M+M;
Q = zeros(n,n);
Q(1:M,1:M) = (sigma_rad^2)*eye(M);
Q(M+(1:M),M+(1:M)) = (sigma_rad^2)*eye(M);
Q(M+M+(1:nd),M+M+(1:nd)) = (sigma_diffD^2)*eye(nd);
Q(M+nd+M+(1:nd), M+nd+M+(1:nd)) = (sigma_groa^2)*eye(nd);
end

function Q = make_Q_AT(M, sigma_diffD, sigma_rad)
% AT z = [d; theta; beta]
nd = M-1;
n = nd + M + M;
Q = zeros(n,n);
Q(1:M,1:M) = (sigma_rad^2)*eye(M);
Q(M+(1:M),M+(1:M)) = (sigma_rad^2)*eye(M);
Q(M+M+(1:nd),M+M+(1:nd)) = (sigma_diffD^2)*eye(nd);
end

function Q = make_Q_AG(M, sigma_rad, sigma_groa)
% AG z = [rho; theta; beta]
nd = M-1;
n = nd + M + M;
Q = zeros(n,n);
Q(1:M,1:M) = (sigma_rad^2)*eye(M);
Q(M+(1:M),M+(1:M)) = (sigma_rad^2)*eye(M);
Q(M+M+(1:nd),M+M+(1:nd)) = (sigma_groa^2)*eye(nd);
end

function Q = make_Q_TG(M, sigma_diffD, sigma_groa)
% TG z = [diff;rho;]
nd = M-1;
n = nd + nd;
Q = zeros(n,n);
Q(1:nd,1:nd) = (sigma_diffD^2)*eye(nd);
Q(nd+(1:nd),nd+(1:nd)) = (sigma_groa^2)*eye(nd);
end