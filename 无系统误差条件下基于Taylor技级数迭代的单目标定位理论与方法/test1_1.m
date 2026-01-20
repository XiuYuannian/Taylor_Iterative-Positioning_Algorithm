%% =====================目标位置估计均方根误差随角度测量量扰动参数的变化曲线======================
clc; clear; close all;
%% ========================场景：站点与目标====================================
u_true = [4,3.8,2]*1e3;
W = [ 0.6,  1.4,  0.2;
    -1.2, -0.6,  0.15;
     1.4, -0.4, -0.2;
    -0.5,  0.8,  0.12;
     1.3, -0.4, -0.25;
    -0.8,  1.0, -0.15] * 1e3;
M = size(W,1);
%% ================== 噪声设置 ==================
sigma_diffD = 2;
deta1 = 1:1:20;
sigma_rad_list = 0.001 * deta1;
sigma_groa = 0.01;

%% ================== Monte Carlo & 迭代设置 ==================
MC = 200;
rng(42);
maxIter = 60;
tol = 1e-6;
u0 = mean(W,1) + [2000, -1500, 800];   % 初值 (m)

%% ================== 存储：RMSE 与 CRB(1D) ==================
RMSE_ATG = zeros(numel(deta1),1);  % AOA+TDOA+GROA
RMSE_AT  = zeros(numel(deta1),1);  % AOA+TDOA
RMSE_AG  = zeros(numel(deta1),1);  % AOA+GROA
CRB_ATG_1D = zeros(numel(deta1),1);
CRB_AT_1D  = zeros(numel(deta1),1);
CRB_AG_1D  = zeros(numel(deta1),1);

%% ================== 预先计算真值观测（无噪） ==================
[theta_t, beta_t, d_t, rho_t] = obs(u_true, W);
%% ================== 主循环：随 AOA 噪声变化 ==================
for k = 1:numel(sigma_rad_list)
    sigma_rad = sigma_rad_list(k);
    % ====== 3种组合的Q(对角阵）======
    Q_ATG = make_Q_ATG(M, sigma_diffD, sigma_rad, sigma_groa);
    Q_AT  = make_Q_AT(M,  sigma_diffD, sigma_rad);
    Q_AG  = make_Q_AG(M,  sigma_rad,   sigma_groa);
    % ====== CRB：在真值处计算（不用MC）======
    CRB_ATG_1D(k) = crb_1d(u_true, W, Q_ATG, "ATG");
    CRB_AT_1D(k)  = crb_1d(u_true, W, Q_AT,  "AT");
    CRB_AG_1D(k)  = crb_1d(u_true, W, Q_AG,  "AG");
    % ====== Monte Carlo：RMSE ======
    err_ATG = zeros(MC,1);
    err_AT  = zeros(MC,1);
    err_AG  = zeros(MC,1);
    for mc = 1:MC
        % ---- 生成一次“完整观测”的噪声，然后按组合取子集 ----
        z_full_true = [theta_t; beta_t;d_t; rho_t];
        Q_full = make_Q_ATG(M, sigma_diffD, sigma_rad, sigma_groa);
        z_full = z_full_true + chol(Q_full,'lower') * randn(length(z_full_true),1);
        % ---- 组合1：ATG
        z_ATG = z_full;                 % 全量
        u_hat_ATG = solve_wls(u0, z_ATG, W, Q_ATG, "ATG", maxIter, tol);
        err_ATG(mc) = norm(u_hat_ATG - u_true);
        % ---- 组合2：AT
        z_AT = [z_full(1:(M)); z_full((M)+ (1:M)); z_full(M+M + (1:M-1)) ];
        u_hat_AT = solve_wls(u0, z_AT, W, Q_AT, "AT", maxIter, tol);
        err_AT(mc) = norm(u_hat_AT - u_true);
        % ---- 组合3：AG = [rho; theta; beta]
        z_AG = [z_full(1:(M)); z_full((M)+ (1:M)); z_full(M+M+M-1 + (1:M-1)) ];
        u_hat_AG = solve_wls(u0, z_AG, W, Q_AG, "AG", maxIter, tol);
        err_AG(mc) = norm(u_hat_AG - u_true);
    end
    RMSE_ATG(k) = sqrt(mean(err_ATG.^2));
    RMSE_AT(k)  = sqrt(mean(err_AT.^2));
    RMSE_AG(k)  = sqrt(mean(err_AG.^2));
    %fprintf('deta1=%2d | RMSE(ATG/AT/AG)=%.3f / %.3f / %.3f m | CRB1D(ATG/AT/AG)=%.3f / %.3f / %.3f m\n',...
        %deta1(k), RMSE_ATG(k), RMSE_AT(k), RMSE_AG(k), CRB_ATG_1D(k), CRB_AT_1D(k), CRB_AG_1D(k));
end

%% ================== 绘图：RMSE ==================
figure; hold on; box on; grid on;
% ===== RMSE（实线）=====
plot(deta1, RMSE_ATG, '-o', 'LineWidth', 1.8);
plot(deta1, RMSE_AT,  '-s', 'LineWidth', 1.8);
plot(deta1, RMSE_AG,  '-^', 'LineWidth', 1.8);
% ===== CRB（虚线）=====
plot(deta1, CRB_ATG_1D, '--o', 'LineWidth', 1.8);
plot(deta1, CRB_AT_1D,  '--s', 'LineWidth', 1.8);
plot(deta1, CRB_AG_1D,  '--^', 'LineWidth', 1.8);
xlabel('\delta_1');
ylabel('Position Error (m)');
title('AOA / TDOA / GROA 联合定位性能对比');
legend( ...
    'RMSE: AOA+TDOA+GROA', ...
    'RMSE: AOA+TDOA', ...
    'RMSE: AOA+GROA', ...
    'CRB:  AOA+TDOA+GROA', ...
    'CRB:  AOA+TDOA', ...
    'CRB:  AOA+GROA', ...
    'Location','northwest');
set(gca,'FontSize',12);


%% ====================函数区===============================================
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