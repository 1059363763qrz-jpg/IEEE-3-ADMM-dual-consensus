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

fprintf('=== ToyCase v7.2 (SuffixAvg Dual): solver=%s, max_iter=%d, tol_pri=%.1e, wall=%.0fs ===\n', ...
    par.solver.name, par.alg.max_iter, par.alg.tol_pri, par.alg.max_walltime);
fprintf('Dual params: alpha0=%.3g, eps=%.3g, step_clip=%.2f, burnin=%d, lam_clip=%.0f\n', ...
    par.alg.alpha0, par.alg.adagrad_eps, par.alg.step_clip, par.alg.burnin, par.alg.lambda_clip);

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
if admm_feas.status==0 && isfinite(admm_feas.obj)
    S.admm_feas_gap_final = (admm_feas.obj - cen.obj) ./ max(1e-9, abs(cen.obj));
else
    S.admm_feas_gap_final = nan;
    warning('[ADMM-feas-eval] infeasible/failed (status=%d, msg=%s); set feas gap to NaN.', admm_feas.status, admm_feas.yalmiperror);
    print_fixed_diagnosis('ADMM', admm_feas);
end
S.admm_comm = admm.comm.total_scalars;
S.admm_r_end = admm.hist.r_pri(end);
S.admm_rd_end = admm.hist.r_dual(end);

S.cons_iter = numel(cons.hist.obj);
S.cons_time = sum(cons.hist.time);
S.cons_gap_final = gap_cons(end);
S.cons_feas_status = cons_feas.status;
S.cons_feas_obj = cons_feas.obj;
if cons_feas.status==0 && isfinite(cons_feas.obj)
    S.cons_feas_gap_final = (cons_feas.obj - cen.obj) ./ max(1e-9, abs(cen.obj));
else
    S.cons_feas_gap_final = nan;
    warning('[CONS-feas-eval] infeasible/failed (status=%d, msg=%s); set feas gap to NaN.', cons_feas.status, cons_feas.yalmiperror);
    print_fixed_diagnosis('CONS', cons_feas);
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
fprintf('ADMM: iter=%d, time=%.2fs, gap(raw)=%.4f%%, gap(feas)=%.4f%%, feas_status=%d, r_pri=%.3e, r_dual=%.3e, comm=%d\n', ...
    S.admm_iter, S.admm_time, 100*S.admm_gap_final, 100*S.admm_feas_gap_final, S.admm_feas_status, S.admm_r_end, S.admm_rd_end, S.admm_comm);
fprintf('Cons: iter=%d, time=%.2fs, gap(raw)=%.4f%%, gap(feas)=%.4f%%, feas_status=%d, r_pri=%.3e, comm=%d\n', ...
    S.cons_iter, S.cons_time, 100*S.cons_gap_final, 100*S.cons_feas_gap_final, S.cons_feas_status, S.cons_r_end, S.cons_comm);
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
    S.admm_iter, S.admm_time, S.admm_gap_final, S.admm_feas_status, S.admm_feas_obj, S.admm_feas_gap_final, S.admm_r_end, S.admm_rd_end, S.admm_comm, ...
    S.dual_iter, S.dual_time, S.dual_r_last_end, S.dual_r_avg_end, S.dual_gap_end, S.dual_g_end, S.dual_step_mean_end, S.dual_step_max_end, S.dual_comm, ...
    S.cons_iter, S.cons_time, S.cons_gap_final, S.cons_feas_status, S.cons_feas_obj, S.cons_feas_gap_final, S.cons_r_end, S.cons_comm, ...
    'VariableNames', {'central_obj','central_time', ...
                      'admm_iter','admm_time','admm_gap_final','admm_feas_status','admm_feas_obj','admm_feas_gap_final','admm_r_pri_end','admm_r_dual_end','admm_comm', ...
                      'dual_iter','dual_time','dual_r_last_end','dual_r_suffixavg_end','dual_dual_gap_end','dual_g_end','dual_step_mean_end','dual_step_max_end','dual_comm', ...
                      'cons_iter','cons_time','cons_gap_final','cons_feas_status','cons_feas_obj','cons_feas_gap_final','cons_r_pri_end','cons_comm'});
writetable(Tsum,'ToyCaseV7_2_Summary.csv');
fprintf('[Saved] ToyCaseV7_2_Report.mat and ToyCaseV7_2_Summary.csv\n');
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
