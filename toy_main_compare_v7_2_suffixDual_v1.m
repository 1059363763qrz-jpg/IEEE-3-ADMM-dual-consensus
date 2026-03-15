function report = toy_main_compare_v7_2_suffixDual_v1()
% Full toycase compare:
%   Centralized benchmark
%   ADMM
%   Dual (Mirror/AdaGrad + suffix averaging)
%   Penalty consensus

clc; close all;
par = toy_build_params_v5();

% Defaults for improved dual (can override in toy_build_params_v5.m)
if ~isfield(par.alg,'alpha0'); par.alg.alpha0 = 0.05; end
if ~isfield(par.alg,'adagrad_eps'); par.alg.adagrad_eps = 1e-3; end
if ~isfield(par.alg,'step_clip'); par.alg.step_clip = 5.0; end
if ~isfield(par.alg,'burnin'); par.alg.burnin = 100; end
if ~isfield(par.alg,'lambda_clip'); par.alg.lambda_clip = 200; end
if ~isfield(par.alg,'dual_momentum'); par.alg.dual_momentum = 0.5; end
if ~isfield(par.alg,'dual_restart_ratio'); par.alg.dual_restart_ratio = 1.05; end
if ~isfield(par.alg,'dual_subgrad_clip'); par.alg.dual_subgrad_clip = 2.0; end
if ~isfield(par.alg,'dual_suffix_weighted'); par.alg.dual_suffix_weighted = true; end

fprintf('=== ToyCase v7.2 (SuffixAvg Dual): solver=%s, max_iter=%d, tol_pri=%.1e, wall=%.0fs ===\n', ...
    par.solver.name, par.alg.max_iter, par.alg.tol_pri, par.alg.max_walltime);
fprintf('Dual params: alpha0=%.3g, eps=%.3g, step_clip=%.2f, burnin=%d, lam_clip=%.0f\n', ...
    par.alg.alpha0, par.alg.adagrad_eps, par.alg.step_clip, par.alg.burnin, par.alg.lambda_clip);
fprintf('Dual accel: mom=%.2f, restart=%.2f, subgrad_clip=%.2f, weighted_suffix=%d\n', ...
    par.alg.dual_momentum, par.alg.dual_restart_ratio, par.alg.dual_subgrad_clip, par.alg.dual_suffix_weighted);
if isfield(par.alg,'feas_terminal_tol')
    fprintf('Feasible re-eval: terminal SOC tolerance (fixed profile) = %.3g\n', par.alg.feas_terminal_tol);
end
if isfield(par.alg,'feas_fixed_tol')
    fprintf('Feasible re-eval: fixed exchange matching tolerance = %.3g\n', par.alg.feas_fixed_tol);
end

fprintf('\n[1/4] Solving CENTRALIZED benchmark...\n');
tic;
cen = toy_solve_centralized_v5(par);
t_cen = toc;
fprintf('[Central] status=%d (%s)\n', cen.status, cen.yalmiperror);
if cen.status~=0
    error('Centralized failed; stop.');
end
fprintf('[Central] obj*=%.6f, time=%.3fs\n', cen.obj, t_cen);
save('ToyCaseV7_2_Central.mat','cen','par','t_cen');

fprintf('\n[2/4] Running ADMM...\n');
admm = toy_run_admm_v5(par);
save('ToyCaseV7_2_ADMM.mat','admm');

fprintf('\n[3/4] Running Dual (Mirror/AdaGrad + suffix avg)...\n');
dual = toy_run_dual_mirror_adagrad_suffixAvg_v1(par, cen.obj);
save('ToyCaseV7_2_Dual.mat','dual');

fprintf('\n[4/4] Running Penalty-consensus...\n');
cons = toy_run_penalty_consensus_v5(par);
save('ToyCaseV7_2_Consensus.mat','cons');

% NOTE:
% admm.hist.obj / cons.hist.obj are local objective sums at (generally)
% infeasible iterates, so they can appear better than the centralized optimum.
% Re-evaluate each method on a globally feasible profile reconstructed from
% final consensus variables z.
admm_fix = struct('dso_c',admm.final.z.dso_c,'dso_d',admm.final.z.dso_d, ...
                  'mg_c',admm.final.z.mg_c,'mg_d',admm.final.z.mg_d,'buy',admm.final.z.buy);
cons_fix = struct('dso_c',cons.final.z.dso_c,'dso_d',cons.final.z.dso_d, ...
                  'mg_c',cons.final.z.mg_c,'mg_d',cons.final.z.mg_d,'buy',cons.final.z.buy);
admm_feas = call_central_with_fixed_compat(par, admm_fix);
cons_feas = call_central_with_fixed_compat(par, cons_fix);

gap_admm = (admm.hist.obj - cen.obj) ./ max(1e-9, abs(cen.obj));
gap_cons = (cons.hist.obj - cen.obj) ./ max(1e-9, abs(cen.obj));

S=struct();
S.central_obj = cen.obj;
S.central_time = t_cen;

S.admm_iter = numel(admm.hist.obj);
S.admm_time = sum(admm.hist.time);
S.admm_gap_final = gap_admm(end);
S.admm_feas_status = admm_feas.status;
S.admm_feas_obj = admm_feas.obj;
S.admm_repair_status = nan;
S.admm_repair_obj = nan;
S.admm_repair_gap_final = nan;
S.admm_repair_dev_l2 = nan;
S.admm_gap_effective = nan;
if admm_feas.status==0 && isfinite(admm_feas.obj)
    S.admm_feas_gap_final = (admm_feas.obj - cen.obj) ./ max(1e-9, abs(cen.obj));
else
    S.admm_feas_gap_final = nan;
    warning('[ADMM-feas-eval] infeasible/failed (status=%d, msg=%s); set feas gap to NaN.', admm_feas.status, admm_feas.yalmiperror);
    print_fixed_diagnosis('ADMM', admm_feas);
    admm_repair = solve_central_soft_fixed(par, admm_fix);
    S.admm_repair_status = admm_repair.status;
    S.admm_repair_obj = admm_repair.obj;
    S.admm_repair_dev_l2 = admm_repair.dev_l2;
    if admm_repair.status==0 && isfinite(admm_repair.obj)
        S.admm_repair_gap_final = (admm_repair.obj - cen.obj) ./ max(1e-9, abs(cen.obj));
    end
end
S.admm_gap_effective = S.admm_feas_gap_final;
if isnan(S.admm_gap_effective) && ~isnan(S.admm_repair_gap_final)
    S.admm_gap_effective = S.admm_repair_gap_final;
end
S.admm_comm = admm.comm.total_scalars;
S.admm_r_end = admm.hist.r_pri(end);
S.admm_rd_end = admm.hist.r_dual(end);

S.cons_iter = numel(cons.hist.obj);
S.cons_time = sum(cons.hist.time);
S.cons_gap_final = gap_cons(end);
S.cons_feas_status = cons_feas.status;
S.cons_feas_obj = cons_feas.obj;
S.cons_repair_status = nan;
S.cons_repair_obj = nan;
S.cons_repair_gap_final = nan;
S.cons_repair_dev_l2 = nan;
S.cons_gap_effective = nan;
if cons_feas.status==0 && isfinite(cons_feas.obj)
    S.cons_feas_gap_final = (cons_feas.obj - cen.obj) ./ max(1e-9, abs(cen.obj));
else
    S.cons_feas_gap_final = nan;
    warning('[CONS-feas-eval] infeasible/failed (status=%d, msg=%s); set feas gap to NaN.', cons_feas.status, cons_feas.yalmiperror);
    print_fixed_diagnosis('CONS', cons_feas);
    cons_repair = solve_central_soft_fixed(par, cons_fix);
    S.cons_repair_status = cons_repair.status;
    S.cons_repair_obj = cons_repair.obj;
    S.cons_repair_dev_l2 = cons_repair.dev_l2;
    if cons_repair.status==0 && isfinite(cons_repair.obj)
        S.cons_repair_gap_final = (cons_repair.obj - cen.obj) ./ max(1e-9, abs(cen.obj));
    end
end
S.cons_gap_effective = S.cons_feas_gap_final;
if isnan(S.cons_gap_effective) && ~isnan(S.cons_repair_gap_final)
    S.cons_gap_effective = S.cons_repair_gap_final;
end
S.cons_comm = cons.comm.total_scalars;
S.cons_r_end = cons.hist.r_pri(end);

S.dual_iter = numel(dual.hist.r_pri_last);
S.dual_time = sum(dual.hist.time);
S.dual_comm = dual.comm.total_scalars;
S.dual_r_last_end = dual.hist.r_pri_last(end);
S.dual_r_avg_end = dual.hist.r_pri_avg(end);
S.dual_gap_end = dual.hist.dual_gap_rel(end);
S.dual_g_end = dual.hist.g_dual(end);
S.dual_step_mean_end = dual.hist.step_mean(end);
S.dual_step_max_end = dual.hist.step_max(end);

fprintf('\n=== Summary ===\n');
fprintf('ADMM: iter=%d, time=%.2fs, gap(raw)=%.4f%%, gap(feas)=%.4f%%, gap(eff)=%.4f%%, feas_status=%d, r_pri=%.3e, r_dual=%.3e, comm=%d\n', ...
    S.admm_iter, S.admm_time, 100*S.admm_gap_final, 100*S.admm_feas_gap_final, 100*S.admm_gap_effective, S.admm_feas_status, S.admm_r_end, S.admm_rd_end, S.admm_comm);
if ~isnan(S.admm_repair_status)
    fprintf('      repair: status=%d, gap(repair)=%.4f%%, dev_l2=%.3e\n', S.admm_repair_status, 100*S.admm_repair_gap_final, S.admm_repair_dev_l2);
end
fprintf('Cons: iter=%d, time=%.2fs, gap(raw)=%.4f%%, gap(feas)=%.4f%%, gap(eff)=%.4f%%, feas_status=%d, r_pri=%.3e, comm=%d\n', ...
    S.cons_iter, S.cons_time, 100*S.cons_gap_final, 100*S.cons_feas_gap_final, 100*S.cons_gap_effective, S.cons_feas_status, S.cons_r_end, S.cons_comm);
if ~isnan(S.cons_repair_status)
    fprintf('      repair: status=%d, gap(repair)=%.4f%%, dev_l2=%.3e\n', S.cons_repair_status, 100*S.cons_repair_gap_final, S.cons_repair_dev_l2);
end
fprintf('Dual(SuffixAvg): iter=%d, time=%.2fs, r_last=%.3e, r_avg=%.3e, dual_gap=%.3g, g=%.3f, step_mean=%.2e, step_max=%.2e, comm=%d\n', ...
    S.dual_iter, S.dual_time, S.dual_r_last_end, S.dual_r_avg_end, S.dual_gap_end, S.dual_g_end, S.dual_step_mean_end, S.dual_step_max_end, S.dual_comm);

figure('Name','Primal residual');
semilogy(admm.hist.r_pri,'LineWidth',1.6); hold on;
semilogy(dual.hist.r_pri_last,'LineWidth',1.6);
semilogy(dual.hist.r_pri_avg,'--','LineWidth',1.6);
semilogy(cons.hist.r_pri,'LineWidth',1.6);
grid on; xlabel('Iteration'); ylabel('Primal residual (2-norm)');
legend('ADMM','Dual (last)','Dual (suffix avg)','Penalty consensus','Location','northeast');

figure('Name','Objective gap (ADMM/Cons)');
plot(gap_admm,'LineWidth',1.6); hold on;
plot(gap_cons,'LineWidth',1.6);
yline(0,'k--');
grid on; xlabel('Iteration'); ylabel('(obj - obj^*) / |obj^*|');
legend('ADMM','Penalty consensus','Location','northeast');

report=struct('par',par,'central',cen,'central_time',t_cen,'admm',admm,'dual',dual,'consensus',cons,'summary',S);
save('ToyCaseV7_2_Report.mat','report');

Tsum = table( ...
    S.central_obj, S.central_time, ...
    S.admm_iter, S.admm_time, S.admm_gap_final, S.admm_feas_status, S.admm_feas_obj, S.admm_feas_gap_final, S.admm_gap_effective, S.admm_repair_status, S.admm_repair_obj, S.admm_repair_gap_final, S.admm_repair_dev_l2, S.admm_r_end, S.admm_rd_end, S.admm_comm, ...
    S.dual_iter, S.dual_time, S.dual_r_last_end, S.dual_r_avg_end, S.dual_gap_end, S.dual_g_end, S.dual_step_mean_end, S.dual_step_max_end, S.dual_comm, ...
    S.cons_iter, S.cons_time, S.cons_gap_final, S.cons_feas_status, S.cons_feas_obj, S.cons_feas_gap_final, S.cons_gap_effective, S.cons_repair_status, S.cons_repair_obj, S.cons_repair_gap_final, S.cons_repair_dev_l2, S.cons_r_end, S.cons_comm, ...
    'VariableNames', {'central_obj','central_time', ...
                      'admm_iter','admm_time','admm_gap_final','admm_feas_status','admm_feas_obj','admm_feas_gap_final','admm_gap_effective','admm_repair_status','admm_repair_obj','admm_repair_gap_final','admm_repair_dev_l2','admm_r_pri_end','admm_r_dual_end','admm_comm', ...
                      'dual_iter','dual_time','dual_r_last_end','dual_r_suffixavg_end','dual_dual_gap_end','dual_g_end','dual_step_mean_end','dual_step_max_end','dual_comm', ...
                      'cons_iter','cons_time','cons_gap_final','cons_feas_status','cons_feas_obj','cons_feas_gap_final','cons_gap_effective','cons_repair_status','cons_repair_obj','cons_repair_gap_final','cons_repair_dev_l2','cons_r_pri_end','cons_comm'});
writetable(Tsum,'ToyCaseV7_2_Summary.csv');
fprintf('[Saved] ToyCaseV7_2_Report.mat and ToyCaseV7_2_Summary.csv\n');
end

function out = solve_central_soft_fixed(par, fixed)
% If exact fixed profile is infeasible, solve a soft-fixed centralized problem.
% Adds quadratic penalties to deviations from fixed exchanges and returns
% objective + total deviation norm for diagnosis.
T = par.T;

% DSO vars
P_grid = sdpvar(1,T); P_G = sdpvar(1,T);
P_dso_charge = sdpvar(1,T); P_dso_discharge = sdpvar(1,T); P_to_mg = sdpvar(1,T);
% SESO vars
P_s_ch=sdpvar(1,T); P_s_dis=sdpvar(1,T); E_s=sdpvar(1,T+1);
P_to_dso_c=sdpvar(1,T); P_to_dso_d=sdpvar(1,T); P_from_mg_c=sdpvar(1,T); P_from_mg_d=sdpvar(1,T);
% MG vars
P_m_self_ch=sdpvar(1,T); P_m_self_dis=sdpvar(1,T); P_m_lease_c=sdpvar(1,T); P_m_lease_d=sdpvar(1,T); P_m_buy=sdpvar(1,T); E_m=sdpvar(1,T+1);

C=[];
C=[C, P_grid>=0, par.P_Gmin<=P_G<=par.P_Gmax];
C=[C, 0<=P_dso_charge<=par.P_dso_charge_max, 0<=P_dso_discharge<=par.P_dso_discharge_max, 0<=P_to_mg<=par.P_mg_buy_max];
for tt=1:T
    C=[C, P_grid(tt)+P_G(tt)+P_dso_discharge(tt)+par.P_R2(tt) == par.P_D2(tt)+par.P_D3(tt)+P_dso_charge(tt)+P_to_mg(tt)];
end

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
if par.seso.Eend_eq_E0, C=[C, E_s(T+1)==par.seso.E0]; end

C=[C, P_m_self_ch>=0, P_m_self_dis>=0, E_m(1)==par.mg.E0];
C=[C, 0<=P_m_lease_c<=par.P_mg_lease_charge_max, 0<=P_m_lease_d<=par.P_mg_lease_discharge_max, 0<=P_m_buy<=par.P_mg_buy_max];
for tt=1:T
    C=[C, par.mg.P_R(tt)+P_m_self_dis(tt)+P_m_buy(tt) == par.mg.P_L(tt)+P_m_self_ch(tt)];
    C=[C, 0<=P_m_self_ch(tt)+P_m_lease_c(tt)<=par.mg.P_ch_max];
    C=[C, 0<=P_m_self_dis(tt)+P_m_lease_d(tt)<=par.mg.P_dis_max];
    C=[C, E_m(tt+1)==E_m(tt)+par.mg.eta_ch*(P_m_self_ch(tt)+P_m_lease_c(tt))-(1/par.mg.eta_dis)*(P_m_self_dis(tt)+P_m_lease_d(tt))];
    C=[C, 0<=E_m(tt)<=par.mg.E_max];
end
C=[C, 0<=E_m(T+1)<=par.mg.E_max];
if par.mg.Eend_eq_E0, C=[C, E_m(T+1)==par.mg.E0]; end

% Consensus equalities
C=[C, P_to_dso_c==P_dso_charge, P_to_dso_d==P_dso_discharge];
C=[C, P_from_mg_c==P_m_lease_c, P_from_mg_d==P_m_lease_d];
C=[C, P_to_mg==P_m_buy];

J_dso  = sum(par.c_grid.*P_grid + par.c_gen*P_G);
J_seso = sum(par.seso.c_ch*P_s_ch + par.seso.c_dis*P_s_dis + par.seso.c_lease*(P_from_mg_c+P_from_mg_d));
J_mg   = sum(par.mg.c_ch*(P_m_self_ch+P_m_lease_c) + par.mg.c_dis*(P_m_self_dis+P_m_lease_d) + par.c_grid.*P_m_buy);
Obj = J_dso + J_seso + J_mg;

% Penalize deviation from fixed profile to get nearest feasible point.
gamma = 1e3;
dev = [P_dso_charge-fixed.dso_c, P_dso_discharge-fixed.dso_d, P_m_lease_c-fixed.mg_c, P_m_lease_d-fixed.mg_d, P_to_mg-fixed.buy];
Obj_soft = Obj + gamma*sum(dev.^2);

ops = toy_sdpsettings_v5(par);
sol = optimize(C, Obj_soft, ops);
out = struct('status',sol.problem,'yalmiperror',yalmiperror(sol.problem),'obj',nan,'dev_l2',nan);
if sol.problem==0
    out.obj = value(Obj);
    out.dev_l2 = norm(value(dev),2);
end
end

function print_fixed_diagnosis(tag, res)
if ~isfield(res, 'fixed_diagnosis') || isempty(res.fixed_diagnosis)
    fprintf('[%s-feas-eval] no fixed_diagnosis available.\n', tag);
    return;
end
d = res.fixed_diagnosis;
if isfield(d,'bounds')
    b = d.bounds;
    fprintf(['[%s-feas-eval] bounds violation check: neg=%.3g, dso_c=%.3g, dso_d=%.3g, ' ...
             'mg_c=%.3g, mg_d=%.3g, buy=%.3g\n'], ...
        tag, b.neg_violation, b.dso_c_max_violation, b.dso_d_max_violation, ...
        b.mg_c_max_violation, b.mg_d_max_violation, b.buy_max_violation);
end
if isfield(d,'derived')
    x = d.derived;
    fprintf(['[%s-feas-eval] derived checks: SESO imp_ch_neg=%.3g imp_dis_neg=%.3g imp_ch_max=%.3g imp_dis_max=%.3g, ' ...
             'SESO soc_min=%.3g soc_max=%.3g terminal_delta=%.3g\n'], ...
        tag, x.seso_imp_ch_neg_violation, x.seso_imp_dis_neg_violation, x.seso_imp_ch_max_violation, x.seso_imp_dis_max_violation, ...
        x.seso_soc_min_violation, x.seso_soc_max_violation, x.seso_terminal_delta);
    fprintf('[%s-feas-eval] derived checks: MG head_ch_neg=%.3g head_dis_neg=%.3g netreq_upper=%.3g netreq_lower=%.3g\n', ...
        tag, x.mg_head_ch_neg_violation, x.mg_head_dis_neg_violation, x.mg_netreq_upper_violation, x.mg_netreq_lower_violation);
end
if isfield(d,'subproblem')
    s = d.subproblem;
    fprintf('[%s-feas-eval] subproblem status: DSO=%d(%s), SESO=%d(%s), MG=%d(%s)\n', ...
        tag, s.dso_status, s.dso_msg, s.seso_status, s.seso_msg, s.mg_status, s.mg_msg);
    if isfield(s,'seso_no_terminal_status')
        fprintf('[%s-feas-eval] SESO without terminal SOC: %d(%s)\n', tag, s.seso_no_terminal_status, s.seso_no_terminal_msg);
    end
    if isfield(s,'mg_no_terminal_status')
        fprintf('[%s-feas-eval] MG without terminal SOC: %d(%s)\n', tag, s.mg_no_terminal_status, s.mg_no_terminal_msg);
    end
end
end

function res = call_central_with_fixed_compat(par, fixed)
% Compatibility wrapper:
% - New signature: toy_solve_centralized_v5(par, fixed)
% - Old signature: toy_solve_centralized_v5(par)
%
% Some environments keep older function versions on MATLAB path or in cache.
% Fallback avoids runtime crash and emits a warning so users can clear path/cache.
try
    res = toy_solve_centralized_v5(par, fixed);
catch ME
    if contains(ME.message, '输入参数太多') || contains(ME.message, 'Too many input arguments')
        warning(['toy_solve_centralized_v5 currently resolves to old 1-input version. ' ...
                 'Falling back to toy_solve_centralized_v5(par). ' ...
                 'Please clear cache/path (e.g., clear functions; rehash path; which -all toy_solve_centralized_v5).']);
        res = toy_solve_centralized_v5(par);
        return;
    end
    rethrow(ME);
end
end
