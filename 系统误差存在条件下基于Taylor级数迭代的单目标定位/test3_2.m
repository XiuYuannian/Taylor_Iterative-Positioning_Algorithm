%% ==================== 目标位置估计均方根误差随系统参量扰动参数的变化曲线（Taylor—b1）======================
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

%% ================== 固定观测噪声设置 ==================
sigma_aoa  = 0.01;   % rad
sigma_tdoa = 0.5;       % “距离差” std（确保与 obs/d_t 单位一致）
sigma_groa = 0.002;    % (确保与 rho_t 单位一致)

%% ================== 扰动参数：站位不确定性随 deta2 变化 ==================
deta2 = 1:20;
sigma_w_list = 0.3 * deta2;      % (m)

%% ================== Monte Carlo & 迭代参数 ==================
MC = 200;
rng(42);
maxIter = 60;
tol = 1e-6;

% 初值（可根据实际进行更改）
u_init = 100 + [2000, -1500, 800];

%% ================== 固定各模式观测协方差矩阵 ==================
Q_ATG = make_Q_ATG(M, sigma_tdoa, sigma_aoa, sigma_groa);
Q_AT  = make_Q_AT(M,  sigma_tdoa, sigma_aoa);
Q_AG  = make_Q_AG(M,  sigma_aoa,  sigma_groa);
Q_TG  = make_Q_TG(M,  sigma_tdoa, sigma_groa);

%% ================== 存储 ==================
RMSE_ATG = zeros(numel(deta2),1);
RMSE_AT  = zeros(numel(deta2),1);
RMSE_AG  = zeros(numel(deta2),1);
RMSE_TG  = zeros(numel(deta2),1);

crb1d_ATG_u = zeros(numel(deta2),1);
crb1d_AT_u  = zeros(numel(deta2),1);
crb1d_AG_u  = zeros(numel(deta2),1);
crb1d_TG_u  = zeros(numel(deta2),1);

crb1d_ATG_w = zeros(numel(deta2),1);
crb1d_AT_w  = zeros(numel(deta2),1);
crb1d_AG_w  = zeros(numel(deta2),1);
crb1d_TG_w  = zeros(numel(deta2),1);

%% ================== 主循环：随 sigma_w 变化 ==================
for k = 1:numel(deta2)
    sigma_w = sigma_w_list(k);          % (m)
    Q2 = (sigma_w^2) * eye(3*M);        % 站位先验/不确定协方差

    % ---- 理论 CRB：每个 k 算一次 ----
    [crb1d_ATG_u(k), crb1d_ATG_w(k)] = crb_1d_taylorb(u_true0, W0, Q_ATG, Q2, "ATG");
    [crb1d_AT_u(k),  crb1d_AT_w(k)]  = crb_1d_taylorb(u_true0, W0, Q_AT,  Q2, "AT");
    [crb1d_AG_u(k),  crb1d_AG_w(k)]  = crb_1d_taylorb(u_true0, W0, Q_AG,  Q2, "AG");
    [crb1d_TG_u(k),  crb1d_TG_w(k)]  = crb_1d_taylorb(u_true0, W0, Q_TG,  Q2, "TG");

    % ---- Monte Carlo ----
    err_ATG = zeros(MC,1);
    err_AT  = zeros(MC,1);
    err_AG  = zeros(MC,1);
    err_TG  = zeros(MC,1);

    for mc = 1:MC
        % 每次 MC 随机真实站位（与 Q2 统计一致）
        ew     = sigma_w * randn(M,3);
        W_true = W0 + ew;

        % 由真实站位产生“真值观测”
        [theta_t, beta_t, d_t, rho_t] = obs(u_true0, W_true);

        % ---- ATG ----  z=[theta; beta; d; rho]
        z_ATG_true = [theta_t; beta_t; d_t; rho_t];
        z_ATG = z_ATG_true + chol(Q_ATG,'lower') * randn(length(z_ATG_true),1);
        [u_hat_ATG, ~] = taylor_b1(u_init, z_ATG, W0, Q_ATG, Q2, "ATG", maxIter, tol);
        err_ATG(mc) = norm(u_hat_ATG - u_true0);

        % ---- AT ---- z=[theta; beta; d]
        z_AT_true = [theta_t; beta_t; d_t];
        z_AT = z_AT_true + chol(Q_AT,'lower') * randn(length(z_AT_true),1);
        [u_hat_AT, ~] = taylor_b1(u_init, z_AT, W0, Q_AT, Q2, "AT", maxIter, tol);
        err_AT(mc) = norm(u_hat_AT - u_true0);

        % ---- AG ---- z=[theta; beta; rho]
        z_AG_true = [theta_t; beta_t; rho_t];
        z_AG = z_AG_true + chol(Q_AG,'lower') * randn(length(z_AG_true),1);
        [u_hat_AG, ~] = taylor_b1(u_init, z_AG, W0, Q_AG, Q2, "AG", maxIter, tol);
        err_AG(mc) = norm(u_hat_AG - u_true0);

        % ---- TG ---- z=[d; rho]
        z_TG_true = [d_t; rho_t];
        z_TG = z_TG_true + chol(Q_TG,'lower') * randn(length(z_TG_true),1);
        [u_hat_TG, ~] = taylor_b1(u_init, z_TG, W0, Q_TG, Q2, "TG", maxIter, tol);
        err_TG(mc) = norm(u_hat_TG - u_true0);
    end

    RMSE_ATG(k) = sqrt(mean(err_ATG.^2));
    RMSE_AT(k)  = sqrt(mean(err_AT.^2));
    RMSE_AG(k)  = sqrt(mean(err_AG.^2));
    RMSE_TG(k)  = sqrt(mean(err_TG.^2));
end

%% ================== 绘图：随 deta2 变化 ==================
figure; grid on; box on; hold on;

plot(deta2, RMSE_ATG, '*', 'LineWidth', 1.8, 'MarkerSize', 6);
plot(deta2, RMSE_AT,  'o', 'LineWidth', 1.8, 'MarkerSize', 6);
plot(deta2, RMSE_AG,  'd', 'LineWidth', 1.8, 'MarkerSize', 6);
plot(deta2, RMSE_TG,  '^', 'LineWidth', 1.8, 'MarkerSize', 6);

plot(deta2, crb1d_ATG_u, '--*','LineWidth', 1.8, 'MarkerSize', 6);
plot(deta2, crb1d_AT_u,  '--o','LineWidth', 1.8, 'MarkerSize', 6);
plot(deta2, crb1d_AG_u,  '--d','LineWidth', 1.8, 'MarkerSize', 6);
plot(deta2, crb1d_TG_u,  '--^','LineWidth', 1.8, 'MarkerSize', 6);

xlabel('\delta_2 ( \sigma_w in m )','FontSize',12);
ylabel('Position Error (m)','FontSize',12);
title('Taylor-b1 RMSE and Hybrid CRB versus \delta_2 (station position uncertainty)', 'FontSize',13);

legend({ ...
    'Taylor-b1-ATG RMSE', ...
    'Taylor-b1-AT RMSE', ...
    'Taylor-b1-AG RMSE', ...
    'Taylor-b1-TG RMSE', ...
    'Hybrid CRB_u (ATG)',...
    'Hybrid CRB_u (AT)',...
    'Hybrid CRB_u (AG)',...
    'Hybrid CRB_u (TG)'}, ...
    'Location','northwest','FontSize',11);

set(gca,'FontSize',12);

%% ======================函数区=================================
function Q = make_Q_ATG(M,sigma_diffD, sigma_rad, sigma_groa)
nd = M-1;
n = M + M + nd + nd;                 % theta(M) + beta(M) + d(nd) + rho(nd)
Q = zeros(n,n);
Q(1:M,1:M) = (sigma_rad^2)*eye(M);                       % theta
Q(M+(1:M),M+(1:M)) = (sigma_rad^2)*eye(M);               % beta
Q(2*M+(1:nd),2*M+(1:nd)) = (sigma_diffD^2)*eye(nd);      % d
Q(2*M+nd+(1:nd),2*M+nd+(1:nd)) = (sigma_groa^2)*eye(nd); % rho
end

function Q = make_Q_AT(M, sigma_diffD, sigma_rad)
nd = M-1;
n = M + M + nd;                      % theta(M) + beta(M) + d(nd)
Q = zeros(n,n);
Q(1:M,1:M) = (sigma_rad^2)*eye(M);
Q(M+(1:M),M+(1:M)) = (sigma_rad^2)*eye(M);
Q(2*M+(1:nd),2*M+(1:nd)) = (sigma_diffD^2)*eye(nd);
end

function Q = make_Q_AG(M, sigma_rad, sigma_groa)
nd = M-1;
n = M + M + nd;                      % theta(M) + beta(M) + rho(nd)
Q = zeros(n,n);
Q(1:M,1:M) = (sigma_rad^2)*eye(M);
Q(M+(1:M),M+(1:M)) = (sigma_rad^2)*eye(M);
Q(2*M+(1:nd),2*M+(1:nd)) = (sigma_groa^2)*eye(nd);
end

function Q = make_Q_TG(M, sigma_diffD, sigma_groa)
nd = M-1;
n = nd + nd;                         % d(nd) + rho(nd)
Q = zeros(n,n);
Q(1:nd,1:nd) = (sigma_diffD^2)*eye(nd);
Q(nd+(1:nd),nd+(1:nd)) = (sigma_groa^2)*eye(nd);
end