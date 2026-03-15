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
    C=[C, E_s(T+1)==par.seso.E0];
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
    C=[C, E_m(T+1)==par.mg.E0];
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
    warning('[Central-feas] status=%d (%s)', sol_feas.problem, yalmiperror(sol_feas.problem));
end
sol = optimize(C, Obj, ops);

res=struct();
res.sol=sol;
res.status=sol.problem;
res.yalmiperror=yalmiperror(sol.problem);
res.obj=value(Obj);
res.J=struct('dso',value(J_dso),'seso',value(J_seso),'mg',value(J_mg));
res.P=struct('dso_c',value(P_dso_charge),'dso_d',value(P_dso_discharge), ...
             'mg_c',value(P_m_lease_c),'mg_d',value(P_m_lease_d),'mg_buy',value(P_m_buy));
end
