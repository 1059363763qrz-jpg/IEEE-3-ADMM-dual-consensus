function out = toy_sweep_dual_stepsize_v1()
% Sweep step-size related parameters for dual-only runs.
% Focus: dual convergence behavior (r_last / r_avg / dual gap / time).

clc;
par0 = toy_build_params_v5();

fprintf('=== Dual sweep v1: solver=%s, max_iter=%d, tol=%.1e ===\n', ...
    par0.solver.name, par0.alg.max_iter, par0.alg.tol_pri);

% Need central optimum as primal upper bound for dual-gap evaluation.
cen = toy_solve_centralized_v5(par0);
if cen.status~=0
    error('Centralized benchmark failed: %s', cen.yalmiperror);
end
fprintf('[Central] obj*=%.6f\n', cen.obj);

% ---- Sweep grid (keep small and practical) ----
alpha_list  = [0.02, 0.03, 0.05];
poly_list   = [0.0, 0.2];
decay_list  = [0.25, 0.4];
mom_list    = [0.0, 0.1];

rows = {};
run_id = 0;

for a = alpha_list
    for pb = poly_list
        for dp = decay_list
            for m = mom_list
                run_id = run_id + 1;

                par = par0;
                par.alg.alpha0 = a;
                par.alg.dual_polyak_blend = pb;
                par.alg.dual_decay_power = dp;
                par.alg.dual_momentum = m;

                % Stabilize sweep settings for comparability
                if ~isfield(par.alg,'dual_decay_warmup'); par.alg.dual_decay_warmup = 50; end
                if ~isfield(par.alg,'dual_restart_ratio'); par.alg.dual_restart_ratio = 1.01; end
                if ~isfield(par.alg,'dual_subgrad_clip'); par.alg.dual_subgrad_clip = 1e6; end
                if ~isfield(par.alg,'dual_suffix_weighted'); par.alg.dual_suffix_weighted = false; end
                if ~isfield(par.alg,'dual_auto_shrink'); par.alg.dual_auto_shrink = true; end
                if ~isfield(par.alg,'dual_patience'); par.alg.dual_patience = 30; end
                if ~isfield(par.alg,'dual_shrink'); par.alg.dual_shrink = 0.7; end
                if ~isfield(par.alg,'dual_min_scale'); par.alg.dual_min_scale = 0.05; end

                fprintf('\n[Run %02d] alpha0=%.3g, polyak=%.2f, decay=%.2f, mom=%.2f\n', run_id, a, pb, dp, m);
                dual = toy_run_dual_mirror_adagrad_suffixAvg_v1(par, cen.obj);

                k = numel(dual.hist.r_pri_last);
                r_last = dual.hist.r_pri_last(end);
                r_avg = dual.hist.r_pri_avg(end);
                g = dual.hist.g_dual(end);
                gap = dual.hist.dual_gap_rel(end);
                t = sum(dual.hist.time);
                step_mean = dual.hist.step_mean(end);
                step_max = dual.hist.step_max(end);
                if isfield(dual.hist,'step_scale')
                    step_scale = dual.hist.step_scale(end);
                else
                    step_scale = nan;
                end

                fprintf('  -> iter=%d, r_last=%.3e, r_avg=%.3e, gap=%.3g, g=%.6f, time=%.1fs\n', ...
                    k, r_last, r_avg, gap, g, t);

                rows(end+1,:) = {run_id, a, pb, dp, m, k, t, r_last, r_avg, gap, g, step_mean, step_max, step_scale}; %#ok<AGROW>
            end
        end
    end
end

T = cell2table(rows, 'VariableNames', { ...
    'run_id','alpha0','polyak_blend','decay_power','momentum', ...
    'iter','time_s','r_last_end','r_avg_end','dual_gap_end','g_dual_end', ...
    'step_mean_end','step_max_end','step_scale_end'});

T = sortrows(T, {'r_avg_end','dual_gap_end'});
writetable(T,'ToyDualSweep_v1.csv');
save('ToyDualSweep_v1.mat','T','rows','par0','cen');

fprintf('\n[Saved] ToyDualSweep_v1.csv / ToyDualSweep_v1.mat\n');
disp(T(1:min(10,height(T)),:));

out = struct('table',T,'central',cen,'par0',par0);
end
