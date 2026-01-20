%% =====================目标位置估计均方根误差随目标距离参数的变化曲线===================
clc; clear; close all;
%% ================== 场景：站点 ==================
u0 = [4, 3.8, 2] * 1e3;   % 原始目标基准位置 (m)
W = [ ...
     0.6,  1.4,  0.2;
    -1.2, -0.6,  0.15;
     1.4, -0.4, -0.2;
    -0.5,  0.8,  0.12;
     1.3, -0.4, -0.25;
    -0.8,  1.0, -0.15] * 1e3;
M = size(W,1);
%% ================== 噪声设置（固定）==================
sigma_diffD = 2;      % TDOA 距离差 std (m)
sigma_rad   = 0.003;  % AOA std (rad)
sigma_groa  = 0.01;   % GROA std (注意：若0.01是方差，则改为 sqrt(0.01)=0.1)
%% ================== 目标平移设置 ==================
d_list = 1:20;
step = 200;           % 每步 200m
dir = [1, 0, 0];      % 平移方向（单位向量）
dir = dir / norm(dir);
%% ================== Monte Carlo & 迭代 ==================
MC = 200;
% rng(42);
maxIter = 60;
tol = 1e-6;
% 初值（每个d都用同一个初值，也可以用上一点的解做 warm-start）
u_init = mean(W,1) + [2000, -1500, 800];
%% ================== 存储：RMSE 与 CRB(1D) ==================
RMSE_ATG = zeros(numel(d_list),1);
RMSE_AT  = zeros(numel(d_list),1);
RMSE_AG  = zeros(numel(d_list),1);
CRB_ATG_1D = zeros(numel(d_list),1);
CRB_AT_1D  = zeros(numel(d_list),1);
CRB_AG_1D  = zeros(numel(d_list),1);
%% ================== 固定Q（因为噪声固定）==================
Q_ATG = make_Q_ATG(M, sigma_diffD, sigma_rad, sigma_groa); % [d; rho; theta; beta]
Q_AT  = make_Q_AT(M,  sigma_diffD, sigma_rad);            % [d; theta; beta]
Q_AG  = make_Q_AG(M,  sigma_rad,   sigma_groa);           % [rho; theta; beta]
Q_full = make_Q_ATG(M, sigma_diffD, sigma_rad, sigma_groa); % 用于一次生成全量噪声
%% ================== 主循环：目标位置变化 ==================
for i = 1:numel(d_list)
    d = d_list(i);
    u_true = u0 + dir * (d * step);    % u_true + d*200m
    % 真值观测（无噪）
    [theta_t, beta_t, dist_t, rho_t] = obs(u_true, W);
    z_full_true = [theta_t; beta_t;dist_t; rho_t];
    Q_full = make_Q_ATG(M, sigma_diffD, sigma_rad, sigma_groa);
    % ---- CRB（真值处）----
    CRB_ATG_1D(i) = crb_1d(u_true, W, Q_ATG, "ATG");
    CRB_AT_1D(i)  = crb_1d(u_true, W, Q_AT,  "AT");
    CRB_AG_1D(i)  = crb_1d(u_true, W, Q_AG,  "AG");
    % ---- Monte Carlo：RMSE ----
    eATG = zeros(MC,1); eAT = zeros(MC,1); eAG = zeros(MC,1);
    for mc = 1:MC
        z_full = z_full_true + chol(Q_full,'lower') * randn(length(z_full_true),1);
        z_ATG = z_full;
        u_hat_ATG = solve_wls(u_init, z_ATG, W, Q_ATG, "ATG", maxIter, tol);
        eATG(mc) = norm(u_hat_ATG - u_true);
    end
    RMSE_ATG(i) = sqrt(mean(eATG.^2));
    fprintf('d=%2d (shift=%4dm) | RMSE(ATG)=%.3f m | CRB1D=%.3f/%.3f/%.3f m\n',...
        d, d*step, RMSE_ATG(i), CRB_ATG_1D(i), CRB_AT_1D(i), CRB_AG_1D(i));
end

%% ================== 绘图：一张图 6 条曲线 ==================
x = d_list * step;  % 横轴用“平移距离(m)”更直观，也可改成 d_list
figure; hold on; grid on; box on;
% RMSE（实线）
plot(x, RMSE_ATG, '-o', 'LineWidth', 1.6);
% CRB（虚线）
plot(x, CRB_ATG_1D, '--o', 'LineWidth', 1.6);
xlabel('Target shift distance (m)');
ylabel('Position error / CRB (m)');
title('RMSE & CRB vs Target Position Shift (AOA/TDOA/GROA combinations)');
legend( ...
    'RMSE: AOA+TDOA+GROA', ...
    'CRB:  AOA+TDOA+GROA', ...
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



