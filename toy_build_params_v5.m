function par = toy_build_params_v5()
% toy_build_params_v5
% v5: MG is allowed to buy power from the distribution network (DSO), making the toy case always feasible.

par = struct();
par.T = 24;
t = 1:par.T;

% --- DSO TOU price (relative) ---
peak   = [8:12, 17:21];
flat   = [6:7, 13:16, 22];
valley = [23,24, 1:5];
par.c_grid = zeros(1,par.T);
par.c_grid(peak)   = 0.80;
par.c_grid(flat)   = 0.60;
par.c_grid(valley) = 0.30;

% --- DSO profiles (MW) from mean values ---
shape_load = 0.90 + 0.18*sin((t-8)/24*2*pi) + 0.08*sin((t-18)/24*2*pi);
shape_load = shape_load/mean(shape_load);
par.P_D2 = 3.5 * shape_load;
par.P_D3 = 1.9 * shape_load;

shape_pv = max(0, sin((t-6)/24*pi*2));
shape_pv = shape_pv / mean(shape_pv(shape_pv>0));
par.P_R2 = 2.0 * 0.55 * shape_pv;

% Dispatchable generator (node3)
par.P_Gmin = 1.0;
par.P_Gmax = 4.0;
par.c_gen  = 0.45;

% DSO requested service bounds (MW)
par.P_dso_charge_max    = 6.0;
par.P_dso_discharge_max = 6.0;

% MG purchase bound from DSO (MW)
par.P_mg_buy_max = 2.5;

% --- SESO storage ---
par.seso.P_ch_max  = 2.5;
par.seso.P_dis_max = 2.5;
par.seso.E_max     = 8.0;
par.seso.eta_ch    = 0.95;
par.seso.eta_dis   = 0.95;
par.seso.E0        = 4.0;
par.seso.Eend_eq_E0 = true;

par.seso.c_ch    = 0.01;
par.seso.c_dis   = 0.01;
par.seso.c_lease = 0.08; % leasing discouraged

% --- MG ---
shape_mg_load = 0.95 + 0.15*sin((t-9)/24*2*pi);
shape_mg_load = shape_mg_load/mean(shape_mg_load);
par.mg.P_L = 1.2 * shape_mg_load;

shape_mg_pv = max(0, sin((t-6)/24*pi*2));
shape_mg_pv = shape_mg_pv / mean(shape_mg_pv(shape_mg_pv>0));
par.mg.P_R = 1.0 * 0.70 * shape_mg_pv; % MW (smaller than load; deficit covered by purchase)

par.mg.P_ch_max  = 1.6;
par.mg.P_dis_max = 1.6;
par.mg.E_max     = 5.0;
par.mg.eta_ch    = 0.95;
par.mg.eta_dis   = 0.95;
par.mg.E0        = 3.0;
par.mg.Eend_eq_E0 = true;

par.mg.c_ch  = 0.01;
par.mg.c_dis = 0.01;

% Leasing limits
par.P_mg_lease_charge_max    = 1.2;
par.P_mg_lease_discharge_max = 1.2;

% --- Algorithm settings ---
par.alg.max_iter = 1000;
par.alg.tol_pri  = 5e-3;
par.alg.tol_dual = 5e-2;
par.alg.print_every = 5;
par.alg.max_walltime = 2400;

par.alg.rho = 5;
par.alg.alpha0 = 0.15;
par.alg.alpha_decay = 'sqrt';
par.alg.beta0 = 1;
par.alg.beta_growth = 1.10;

% Solver
par.solver.name = 'gurobi'; % or 'mosek'
par.solver.verbose = 0;
par.solver.timelimit = 10;

par.alg.alpha0 = 0.05;      % Dual base step
par.alg.adagrad_eps = 1e-3; % 防止极端步长
par.alg.step_clip = 5.0;    % 步长上限
par.alg.burnin = 100;       % 尾平均从第100次开始（你可以试 50/100/200）
par.alg.dual_momentum = 0.15;         % 对偶动量（Heavy-ball，保守默认）
par.alg.dual_restart_ratio = 1.01;    % 残差轻微恶化即重启动量
par.alg.dual_subgrad_clip = 1e6;      % 默认几乎不裁剪（接近原版）
par.alg.dual_suffix_weighted = false; % 默认使用原版等权尾平均
par.alg.dual_polyak_blend = 0.35;     % AdaGrad与Polyak步长混合系数 [0,1]
par.alg.dual_decay_power = 0.25;      % 全局步长衰减幂次（抑制后期振荡）
par.alg.dual_decay_warmup = 50;       % 衰减warmup轮次
par.alg.dual_auto_shrink = true;      % r_avg停滞时自动缩步长
par.alg.dual_patience = 30;           % 停滞判定窗口（仅burnin后）
par.alg.dual_shrink = 0.7;            % 每次自动缩步长系数
par.alg.dual_min_scale = 0.05;        % 自动缩步长最小全局比例
par.alg.feas_terminal_tol = 3e-2; % 可行性回灌评估时，终端SOC容差（仅 fixed 评估使用）
par.alg.feas_fixed_tol = 1e-3;    % 可行性回灌评估时，fixed交换功率匹配容差
end
