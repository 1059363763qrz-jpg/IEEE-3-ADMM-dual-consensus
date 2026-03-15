function res = toy_solve_centralized_v5(par, fixed)
% Centralized benchmark solve for v5 (MG can buy from DSO)

if nargin < 2
    fixed = [];
end

T = par.T;

% DSO vars
P_grid = sdpvar(1,T);
P_G    = sdpvar(1,T);
P_dso_charge    = sdpvar(1,T);
P_dso_discharge = sdpvar(1,T);
P_to_mg = sdpvar(1,T); % DSO supplies MG purchase

% SESO vars
P_s_ch  = sdpvar(1,T);
P_s_dis = sdpvar(1,T);
E_s     = sdpvar(1,T+1);
P_to_dso_c = sdpvar(1,T);
P_to_dso_d = sdpvar(1,T);
P_from_mg_c = sdpvar(1,T);
P_from_mg_d = sdpvar(1,T);

% MG vars
P_m_self_ch  = sdpvar(1,T);
P_m_self_dis = sdpvar(1,T);
P_m_lease_c  = sdpvar(1,T);
P_m_lease_d  = sdpvar(1,T);
P_m_buy      = sdpvar(1,T); % MG purchase from DSO
E_m = sdpvar(1,T+1);

C = [];

% DSO constraints
C=[C, P_grid>=0, par.P_Gmin<=P_G<=par.P_Gmax];
C=[C, 0<=P_dso_charge<=par.P_dso_charge_max, 0<=P_dso_discharge<=par.P_dso_discharge_max];
C=[C, 0<=P_to_mg<=par.P_mg_buy_max];
for tt=1:T
    C=[C, P_grid(tt)+P_G(tt)+P_dso_discharge(tt)+par.P_R2(tt) == par.P_D2(tt)+par.P_D3(tt)+P_dso_charge(tt)+P_to_mg(tt)];
end

% SESO storage & pool
C=[C, P_s_ch>=0, P_s_dis>=0, E_s(1)==par.seso.E0];
C=[C, 0<=P_from_mg_c<=par.P_mg_lease_charge_max, 0<=P_from_mg_d<=par.P_mg_lease_discharge_max];
C=[C, 0<=P_to_dso_c<=par.P_dso_charge_max, 0<=P_to_dso_d<=par.P_dso_discharge_max];
for tt=1:T
    C=[C, E_s(tt+1)==E_s(tt)+par.seso.eta_ch*P_s_ch(tt)-(1/par.seso.eta_dis)*P_s_dis(tt)];
    C=[C, 0<=E_s(tt)<=par.seso.E_max, 0<=P_s_ch(tt)<=par.seso.P_ch_max, 0<=P_s_dis(tt)<=par.seso.P_dis_max];
    C=[C, P_from_mg_c(tt)+P_s_ch(tt)==P_to_dso_c(tt)];
    C=[C, P_from_mg_d(tt)+P_s_dis(tt)==P_to_dso_d(tt)];
end
C=[C, 0<=E_s(T+1)<=par.seso.E_max];
if par.seso.Eend_eq_E0
    tol_end = 0;
    if ~isempty(fixed) && isfield(par,'alg') && isfield(par.alg,'feas_terminal_tol')
        tol_end = par.alg.feas_terminal_tol;
    end
    if tol_end>0
        C=[C, -tol_end <= E_s(T+1)-par.seso.E0 <= tol_end];
    else
        C=[C, E_s(T+1)==par.seso.E0];
    end
end

% MG self-first + buy
C=[C, P_m_self_ch>=0, P_m_self_dis>=0, E_m(1)==par.mg.E0];
C=[C, 0<=P_m_lease_c<=par.P_mg_lease_charge_max, 0<=P_m_lease_d<=par.P_mg_lease_discharge_max];
C=[C, 0<=P_m_buy<=par.P_mg_buy_max];
for tt=1:T
    % MG balance: PV + self_dis + buy = load + self_ch
    C=[C, par.mg.P_R(tt)+P_m_self_dis(tt)+P_m_buy(tt) == par.mg.P_L(tt)+P_m_self_ch(tt)];
    % battery power (self+lease)
    C=[C, 0<=P_m_self_ch(tt)+P_m_lease_c(tt)<=par.mg.P_ch_max];
    C=[C, 0<=P_m_self_dis(tt)+P_m_lease_d(tt)<=par.mg.P_dis_max];
    % SOC update includes self+lease
    C=[C, E_m(tt+1)==E_m(tt)+par.mg.eta_ch*(P_m_self_ch(tt)+P_m_lease_c(tt)) ...
                  -(1/par.mg.eta_dis)*(P_m_self_dis(tt)+P_m_lease_d(tt))];
    C=[C, 0<=E_m(tt)<=par.mg.E_max];
end
C=[C, 0<=E_m(T+1)<=par.mg.E_max];
if par.mg.Eend_eq_E0
    tol_end = 0;
    if ~isempty(fixed) && isfield(par,'alg') && isfield(par.alg,'feas_terminal_tol')
        tol_end = par.alg.feas_terminal_tol;
    end
    if tol_end>0
        C=[C, -tol_end <= E_m(T+1)-par.mg.E0 <= tol_end];
    else
        C=[C, E_m(T+1)==par.mg.E0];
    end
end

% Consensus equalities
C=[C, P_to_dso_c==P_dso_charge, P_to_dso_d==P_dso_discharge];
C=[C, P_from_mg_c==P_m_lease_c, P_from_mg_d==P_m_lease_d];
C=[C, P_to_mg==P_m_buy];

% Optional: fix exchanged powers to a provided consensus profile.
% This is useful for evaluating a globally feasible primal objective recovered
% from distributed methods.
if ~isempty(fixed)
    req = {'dso_c','dso_d','mg_c','mg_d','buy'};
    for i = 1:numel(req)
        if ~isfield(fixed, req{i})
            error('Missing fixed.%s in optional fixed exchange profile.', req{i});
        end
    end
    C=[C, P_dso_charge == fixed.dso_c, P_dso_discharge == fixed.dso_d];
    C=[C, P_m_lease_c  == fixed.mg_c,  P_m_lease_d  == fixed.mg_d];
    C=[C, P_to_mg      == fixed.buy];
end

% Objective (Stage-1 style)
J_dso  = sum(par.c_grid.*P_grid + par.c_gen*P_G);
J_seso = sum(par.seso.c_ch*P_s_ch + par.seso.c_dis*P_s_dis + par.seso.c_lease*(P_from_mg_c+P_from_mg_d));
J_mg   = sum(par.mg.c_ch*(P_m_self_ch+P_m_lease_c) + par.mg.c_dis*(P_m_self_dis+P_m_lease_d) + par.c_grid.*P_m_buy);
Obj = J_dso + J_seso + J_mg;

ops = toy_sdpsettings_v5(par);
sol_feas = optimize(C, 0, ops);
if sol_feas.problem~=0
    if isempty(fixed)
        warning('[Central-feas] status=%d (%s)', sol_feas.problem, yalmiperror(sol_feas.problem));
    else
        warning('[Central-feas-fixed] status=%d (%s)', sol_feas.problem, yalmiperror(sol_feas.problem));
    end
end
sol = optimize(C, Obj, ops);

res=struct();
res.sol=sol;
res.status=sol.problem;
res.yalmiperror=yalmiperror(sol.problem);
res.feas_status=sol_feas.problem;
res.feas_yalmiperror=yalmiperror(sol_feas.problem);
res.fixed_diagnosis=[];
if sol.problem==0
    res.obj=value(Obj);
    res.J=struct('dso',value(J_dso),'seso',value(J_seso),'mg',value(J_mg));
    res.P=struct('dso_c',value(P_dso_charge),'dso_d',value(P_dso_discharge), ...
                 'mg_c',value(P_m_lease_c),'mg_d',value(P_m_lease_d),'mg_buy',value(P_m_buy));
else
    % If solve failed/infeasible, keep outputs explicit to prevent misleading gap values.
    res.obj=nan;
    res.J=struct('dso',nan,'seso',nan,'mg',nan);
    res.P=struct('dso_c',nan(1,T),'dso_d',nan(1,T), ...
                 'mg_c',nan(1,T),'mg_d',nan(1,T),'mg_buy',nan(1,T));
    if ~isempty(fixed)
        res.fixed_diagnosis = diagnose_fixed_profile(par, fixed);
    end
end
end

function diag = diagnose_fixed_profile(par, fixed)
% Diagnose why fixed exchange profile can be infeasible in centralized re-eval.
T = par.T;
ops = toy_sdpsettings_v5(par);

diag = struct();
diag.bounds = struct();
diag.derived = struct();
diag.subproblem = struct();

% -------- Bound checks --------
diag.bounds.dso_c_max_violation = max(0, max(fixed.dso_c - par.P_dso_charge_max));
diag.bounds.dso_d_max_violation = max(0, max(fixed.dso_d - par.P_dso_discharge_max));
diag.bounds.mg_c_max_violation  = max(0, max(fixed.mg_c  - par.P_mg_lease_charge_max));
diag.bounds.mg_d_max_violation  = max(0, max(fixed.mg_d  - par.P_mg_lease_discharge_max));
diag.bounds.buy_max_violation   = max(0, max(fixed.buy   - par.P_mg_buy_max));
diag.bounds.neg_violation = max(0, -min([fixed.dso_c(:); fixed.dso_d(:); fixed.mg_c(:); fixed.mg_d(:); fixed.buy(:)]));

% -------- Derived checks (often root cause) --------
% SESO implied battery powers are fully determined by fixed exchanges:
%   P_s_ch = dso_c - mg_c,  P_s_dis = dso_d - mg_d
imp_s_ch  = fixed.dso_c - fixed.mg_c;
imp_s_dis = fixed.dso_d - fixed.mg_d;
diag.derived.seso_imp_ch_neg_violation = max(0, -min(imp_s_ch));
diag.derived.seso_imp_dis_neg_violation = max(0, -min(imp_s_dis));
diag.derived.seso_imp_ch_max_violation = max(0, max(imp_s_ch - par.seso.P_ch_max));
diag.derived.seso_imp_dis_max_violation = max(0, max(imp_s_dis - par.seso.P_dis_max));

E = zeros(1,T+1); E(1)=par.seso.E0;
for tt=1:T
    E(tt+1) = E(tt) + par.seso.eta_ch*imp_s_ch(tt) - (1/par.seso.eta_dis)*imp_s_dis(tt);
end
diag.derived.seso_soc_min_violation = max(0, -min(E));
diag.derived.seso_soc_max_violation = max(0, max(E - par.seso.E_max));
diag.derived.seso_terminal_delta = E(end) - par.seso.E0;

% MG per-step net balance requirement for self battery:
%   P_self_dis - P_self_ch = P_L - P_R - buy
% Feasibility requires this net demand fits remaining self charge/discharge headroom.
net_req = par.mg.P_L - par.mg.P_R - fixed.buy;
head_ch  = par.mg.P_ch_max  - fixed.mg_c; % max P_self_ch
head_dis = par.mg.P_dis_max - fixed.mg_d; % max P_self_dis
diag.derived.mg_head_ch_neg_violation = max(0, -min(head_ch));
diag.derived.mg_head_dis_neg_violation = max(0, -min(head_dis));
diag.derived.mg_netreq_upper_violation = max(0, max(net_req - head_dis));
diag.derived.mg_netreq_lower_violation = max(0, max((-net_req) - head_ch));

% -------- DSO feasibility under fixed exchange --------
P_grid = sdpvar(1,T); P_G = sdpvar(1,T);
C = [P_grid>=0, par.P_Gmin<=P_G<=par.P_Gmax];
for tt=1:T
    C=[C, P_grid(tt)+P_G(tt)+fixed.dso_d(tt)+par.P_R2(tt) == par.P_D2(tt)+par.P_D3(tt)+fixed.dso_c(tt)+fixed.buy(tt)];
end
sol_dso = optimize(C, 0, ops);
diag.subproblem.dso_status = sol_dso.problem;
diag.subproblem.dso_msg = yalmiperror(sol_dso.problem);

% -------- SESO feasibility under fixed exchange --------
P_s_ch = sdpvar(1,T); P_s_dis = sdpvar(1,T); E_s = sdpvar(1,T+1);
C = [P_s_ch>=0, P_s_dis>=0, E_s(1)==par.seso.E0];
for tt=1:T
    C=[C, E_s(tt+1)==E_s(tt)+par.seso.eta_ch*P_s_ch(tt)-(1/par.seso.eta_dis)*P_s_dis(tt)];
    C=[C, 0<=E_s(tt)<=par.seso.E_max, 0<=P_s_ch(tt)<=par.seso.P_ch_max, 0<=P_s_dis(tt)<=par.seso.P_dis_max];
    C=[C, fixed.mg_c(tt)+P_s_ch(tt)==fixed.dso_c(tt)];
    C=[C, fixed.mg_d(tt)+P_s_dis(tt)==fixed.dso_d(tt)];
end
C=[C, 0<=E_s(T+1)<=par.seso.E_max];
if par.seso.Eend_eq_E0
    C=[C, E_s(T+1)==par.seso.E0];
end
sol_seso = optimize(C, 0, ops);
diag.subproblem.seso_status = sol_seso.problem;
diag.subproblem.seso_msg = yalmiperror(sol_seso.problem);

% SESO check without terminal equality (to isolate end-SOC cause)
if par.seso.Eend_eq_E0
    C_relax = [P_s_ch>=0, P_s_dis>=0, E_s(1)==par.seso.E0];
    for tt=1:T
        C_relax=[C_relax, E_s(tt+1)==E_s(tt)+par.seso.eta_ch*P_s_ch(tt)-(1/par.seso.eta_dis)*P_s_dis(tt)];
        C_relax=[C_relax, 0<=E_s(tt)<=par.seso.E_max, 0<=P_s_ch(tt)<=par.seso.P_ch_max, 0<=P_s_dis(tt)<=par.seso.P_dis_max];
        C_relax=[C_relax, fixed.mg_c(tt)+P_s_ch(tt)==fixed.dso_c(tt)];
        C_relax=[C_relax, fixed.mg_d(tt)+P_s_dis(tt)==fixed.dso_d(tt)];
    end
    C_relax=[C_relax, 0<=E_s(T+1)<=par.seso.E_max];
    sol_seso_relax = optimize(C_relax, 0, ops);
    diag.subproblem.seso_no_terminal_status = sol_seso_relax.problem;
    diag.subproblem.seso_no_terminal_msg = yalmiperror(sol_seso_relax.problem);
end

% -------- MG feasibility under fixed lease/buy --------
P_m_self_ch = sdpvar(1,T); P_m_self_dis = sdpvar(1,T); E_m = sdpvar(1,T+1);
C=[P_m_self_ch>=0, P_m_self_dis>=0, E_m(1)==par.mg.E0];
for tt=1:T
    C=[C, par.mg.P_R(tt)+P_m_self_dis(tt)+fixed.buy(tt) == par.mg.P_L(tt)+P_m_self_ch(tt)];
    C=[C, 0<=P_m_self_ch(tt)+fixed.mg_c(tt)<=par.mg.P_ch_max];
    C=[C, 0<=P_m_self_dis(tt)+fixed.mg_d(tt)<=par.mg.P_dis_max];
    C=[C, E_m(tt+1)==E_m(tt)+par.mg.eta_ch*(P_m_self_ch(tt)+fixed.mg_c(tt)) ...
                  -(1/par.mg.eta_dis)*(P_m_self_dis(tt)+fixed.mg_d(tt))];
    C=[C, 0<=E_m(tt)<=par.mg.E_max];
end
C=[C, 0<=E_m(T+1)<=par.mg.E_max];
if par.mg.Eend_eq_E0
    C=[C, E_m(T+1)==par.mg.E0];
end
sol_mg = optimize(C, 0, ops);
diag.subproblem.mg_status = sol_mg.problem;
diag.subproblem.mg_msg = yalmiperror(sol_mg.problem);

% MG check without terminal equality (to isolate end-SOC cause)
if par.mg.Eend_eq_E0
    C_relax=[P_m_self_ch>=0, P_m_self_dis>=0, E_m(1)==par.mg.E0];
    for tt=1:T
        C_relax=[C_relax, par.mg.P_R(tt)+P_m_self_dis(tt)+fixed.buy(tt) == par.mg.P_L(tt)+P_m_self_ch(tt)];
        C_relax=[C_relax, 0<=P_m_self_ch(tt)+fixed.mg_c(tt)<=par.mg.P_ch_max];
        C_relax=[C_relax, 0<=P_m_self_dis(tt)+fixed.mg_d(tt)<=par.mg.P_dis_max];
        C_relax=[C_relax, E_m(tt+1)==E_m(tt)+par.mg.eta_ch*(P_m_self_ch(tt)+fixed.mg_c(tt)) ...
                              -(1/par.mg.eta_dis)*(P_m_self_dis(tt)+fixed.mg_d(tt))];
        C_relax=[C_relax, 0<=E_m(tt)<=par.mg.E_max];
    end
    C_relax=[C_relax, 0<=E_m(T+1)<=par.mg.E_max];
    sol_mg_relax = optimize(C_relax, 0, ops);
    diag.subproblem.mg_no_terminal_status = sol_mg_relax.problem;
    diag.subproblem.mg_no_terminal_msg = yalmiperror(sol_mg_relax.problem);
end
end
