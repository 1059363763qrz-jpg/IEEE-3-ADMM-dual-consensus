
function report = toy_main_compare_v7_mirrorDual_v1()
% toy_main_compare_v7_mirrorDual_v1
% Full end-to-end toycase comparing:
%   Centralized benchmark
%   ADMM
%   Dual decomposition (Mirror descent / AdaGrad scaling)
%   Penalty-consensus
%
% Outputs:
%   ToyCaseV7_Report.mat
%   ToyCaseV7_Summary.csv

clc; close all;
par = toy_build_params_v5();

% reasonable defaults for new dual if not in params
if ~isfield(par.alg,'alpha0'); par.alg.alpha0 = 0.05; end
if ~isfield(par.alg,'adagrad_eps'); par.alg.adagrad_eps = 1e-6; end
if ~isfield(par.alg,'lambda_clip'); par.alg.lambda_clip = 200; end

fprintf('=== ToyCase v7 (Mirror/AdaGrad Dual): solver=%s, max_iter=%d, tol_pri=%.1e, wall=%.0fs ===\n', ...
    par.solver.name, par.alg.max_iter, par.alg.tol_pri, par.alg.max_walltime);

fprintf('\n[1/4] Solving CENTRALIZED benchmark...\n');
tic;
cen = toy_solve_centralized_v5(par);
t_cen = toc;
fprintf('[Central] status=%d (%s)\n', cen.status, cen.yalmiperror);
if cen.status~=0
    error('Centralized failed; stop.');
end
fprintf('[Central] obj*=%.6f, time=%.3fs\n', cen.obj, t_cen);
save('ToyCaseV7_Central.mat','cen','par','t_cen');

fprintf('\n[2/4] Running ADMM...\n');
admm = toy_run_admm_v5(par);
save('ToyCaseV7_ADMM.mat','admm');

fprintf('\n[3/4] Running Dual decomposition (Mirror/AdaGrad)...\n');
dual = toy_run_dual_mirror_adagrad_v1(par, cen.obj);
save('ToyCaseV7_Dual.mat','dual');

fprintf('\n[4/4] Running Penalty-consensus...\n');
cons = toy_run_penalty_consensus_v5(par);
save('ToyCaseV7_Consensus.mat','cons');

% objective gaps for methods that produce feasible-ish solutions (ADMM/cons)
gap_admm = (admm.hist.obj - cen.obj) ./ max(1e-9, abs(cen.obj));
gap_cons = (cons.hist.obj - cen.obj) ./ max(1e-9, abs(cen.obj));

% Summary struct
S=struct();
S.central_obj = cen.obj;
S.central_time = t_cen;

S.admm_iter = numel(admm.hist.obj);
S.admm_time = sum(admm.hist.time);
S.admm_gap_final = gap_admm(end);
S.admm_comm = admm.comm.total_scalars;
S.admm_r_end = admm.hist.r_pri(end);
S.admm_rd_end = admm.hist.r_dual(end);

S.cons_iter = numel(cons.hist.obj);
S.cons_time = sum(cons.hist.time);
S.cons_gap_final = gap_cons(end);
S.cons_comm = cons.comm.total_scalars;
S.cons_r_end = cons.hist.r_pri(end);

S.dual_iter = numel(dual.hist.r_pri_avg);
S.dual_time = sum(dual.hist.time);
S.dual_comm = dual.comm.total_scalars;
S.dual_r_last_end = dual.hist.r_pri_last(end);
S.dual_r_avg_end = dual.hist.r_pri_avg(end);
S.dual_gap_end = dual.hist.dual_gap_rel(end);
S.dual_g_end = dual.hist.g_dual(end);

fprintf('\n=== Summary (relative to centralized benchmark) ===\n');
fprintf('ADMM: iter=%d, time=%.2fs, gap=%.4f%%, r_pri=%.3e, r_dual=%.3e, comm=%d\n', ...
    S.admm_iter, S.admm_time, 100*S.admm_gap_final, S.admm_r_end, S.admm_rd_end, S.admm_comm);
fprintf('Cons: iter=%d, time=%.2fs, gap=%.4f%%, r_pri=%.3e, comm=%d\n', ...
    S.cons_iter, S.cons_time, 100*S.cons_gap_final, S.cons_r_end, S.cons_comm);
fprintf('Dual(Mirror/AdaGrad): iter=%d, time=%.2fs, r_last=%.3e, r_avg=%.3e, dual_gap=%.3g, g=%.3f, comm=%d\n', ...
    S.dual_iter, S.dual_time, S.dual_r_last_end, S.dual_r_avg_end, S.dual_gap_end, S.dual_g_end, S.dual_comm);

% Plots
figure('Name','Primal residual');
semilogy(admm.hist.r_pri,'LineWidth',1.6); hold on;
semilogy(dual.hist.r_pri_last,'LineWidth',1.6);
semilogy(dual.hist.r_pri_avg,'--','LineWidth',1.6);
semilogy(cons.hist.r_pri,'LineWidth',1.6);
grid on; xlabel('Iteration'); ylabel('Primal residual (2-norm)');
legend('ADMM','Dual (last)','Dual (avg)','Penalty consensus','Location','northeast');

figure('Name','Objective gap (ADMM/Cons)');
plot(gap_admm,'LineWidth',1.6); hold on;
plot(gap_cons,'LineWidth',1.6);
yline(0,'k--');
grid on; xlabel('Iteration'); ylabel('(obj - obj^*) / |obj^*|');
legend('ADMM','Penalty consensus','Location','northeast');

report=struct('par',par,'central',cen,'central_time',t_cen,'admm',admm,'dual',dual,'consensus',cons,'summary',S);
save('ToyCaseV7_Report.mat','report');

Tsum = table( ...
    S.central_obj, S.central_time, ...
    S.admm_iter, S.admm_time, S.admm_gap_final, S.admm_r_end, S.admm_rd_end, S.admm_comm, ...
    S.dual_iter, S.dual_time, S.dual_r_last_end, S.dual_r_avg_end, S.dual_gap_end, S.dual_g_end, S.dual_comm, ...
    S.cons_iter, S.cons_time, S.cons_gap_final, S.cons_r_end, S.cons_comm, ...
    'VariableNames', {'central_obj','central_time', ...
                      'admm_iter','admm_time','admm_gap_final','admm_r_pri_end','admm_r_dual_end','admm_comm', ...
                      'dual_iter','dual_time','dual_r_last_end','dual_r_avg_end','dual_dual_gap_end','dual_g_end','dual_comm', ...
                      'cons_iter','cons_time','cons_gap_final','cons_r_pri_end','cons_comm'});
writetable(Tsum,'ToyCaseV7_Summary.csv');
fprintf('[Saved] ToyCaseV7_Report.mat and ToyCaseV7_Summary.csv\n');
end
