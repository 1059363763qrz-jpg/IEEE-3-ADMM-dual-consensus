
function out = toy_run_dual_mirror_adagrad_v1(par, primal_ub)
% toy_run_dual_mirror_adagrad_v1
% Distributed dual decomposition using mirror descent / AdaGrad diagonal scaling.
% No augmented Lagrangian (NO ADMM). Designed to improve robustness vs plain subgradient.
%
% Key ideas:
%   lambda <- lambda + alpha * s ./ (sqrt(G) + eps)   where G accumulates s.^2
%   Track both:
%     r_last: last-iterate primal residual (may not converge)
%     r_avg : ergodic-averaged primal residual (should decrease in convex settings)
%   Also compute dual function value g(lambda) (lower bound on primal optimum).
%
% Inputs:
%   primal_ub: primal upper bound, typically centralized obj* (used only for reporting dual gap)
%
% Outputs in out.hist:
%   r_pri_last, r_pri_avg, g_dual, dual_gap_rel, alpha_eff, time
%
% Notes:
%   - Correct Lagrangian signs:
%       constraint is (DSO - SESO)=0 and (MG - SESO)=0 and (DSO_buy - MG_buy)=0
%       => DSO uses +lambda, SESO uses -lambda, MG uses +lambda for lease and -lambda for buy.
%   - If you want faster convergence, tune:
%       par.alg.alpha0 (global base stepsize)
%       par.alg.adagrad_eps
%       par.alg.lambda_clip

T = par.T;
K = par.alg.max_iter;
pe = par.alg.print_every;

% Multipliers (same dimension as coupling vectors)
lam_dso_c = zeros(1,T); lam_dso_d = zeros(1,T);
lam_mg_c  = zeros(1,T); lam_mg_d  = zeros(1,T);
lam_buy   = zeros(1,T);

% AdaGrad accumulators
G_dso_c = zeros(1,T); G_dso_d = zeros(1,T);
G_mg_c  = zeros(1,T); G_mg_d  = zeros(1,T);
G_buy   = zeros(1,T);

% Hyperparameters
alpha0 = par.alg.alpha0;              % base stepsize
eps0   = 1e-6;
if isfield(par.alg,'adagrad_eps'); eps0 = par.alg.adagrad_eps; end
LAM_CLIP = 200;
if isfield(par.alg,'lambda_clip'); LAM_CLIP = par.alg.lambda_clip; end

% Ergodic averages for primal recovery
avg_dso_c=zeros(1,T); avg_dso_d=zeros(1,T);
avg_seso_c=zeros(1,T); avg_seso_d=zeros(1,T);
avg_mg_c=zeros(1,T);  avg_mg_d=zeros(1,T);
avg_seso_mg_c=zeros(1,T); avg_seso_mg_d=zeros(1,T);
avg_buy_dso=zeros(1,T); avg_buy_mg=zeros(1,T);

hist.r_pri_last=zeros(K,1);
hist.r_pri_avg=zeros(K,1);
hist.g_dual=zeros(K,1);
hist.dual_gap_rel=zeros(K,1);
hist.alpha_eff=zeros(K,1); % average effective stepsize scale
hist.time=zeros(K,1);

t0=tic;
for k=1:K
    it=tic;

    % Solve decomposed subproblems (dual mode)
    dso  = Fun_DSO_Toy_v5(par, [], [], [], lam_dso_c, lam_dso_d, lam_buy, 0, 'dual');
    seso = Fun_SESO_Toy_v5(par, [], [], -lam_dso_c, -lam_dso_d, 0, [], [], -lam_mg_c, -lam_mg_d, 0, 'dual');
    mg   = Fun_MG_Toy_v5(par, [], [], [], +lam_mg_c, +lam_mg_d, -lam_buy, 0, 'dual');

    if dso.status~=0 || seso.status~=0 || mg.status~=0
        warning('[Dual-AdaGrad] subproblem failed at k=%d (dso=%d seso=%d mg=%d).', k, dso.status, seso.status, mg.status);
        break;
    end

    % Residuals (subgradients for equality constraints)
    s_dso_c = dso.P_charge    - seso.P_to_dso_charge;
    s_dso_d = dso.P_discharge - seso.P_to_dso_discharge;
    s_mg_c  = mg.P_lease_charge    - seso.P_from_mg_charge;
    s_mg_d  = mg.P_lease_discharge - seso.P_from_mg_discharge;
    s_buy   = dso.P_to_mg - mg.P_buy;

    r_last = max([norm(s_dso_c,2), norm(s_dso_d,2), norm(s_mg_c,2), norm(s_mg_d,2), norm(s_buy,2)]);

    % Dual function value g(lambda) = sum_i min_x L_i(x,lambda)
    L_dso  = dso.J  + sum(lam_dso_c.*dso.P_charge) + sum(lam_dso_d.*dso.P_discharge) + sum(lam_buy.*dso.P_to_mg);
    L_seso = seso.J - sum(lam_dso_c.*seso.P_to_dso_charge) - sum(lam_dso_d.*seso.P_to_dso_discharge) ...
                    - sum(lam_mg_c.*seso.P_from_mg_charge) - sum(lam_mg_d.*seso.P_from_mg_discharge);
    L_mg   = mg.J   + sum(lam_mg_c.*mg.P_lease_charge) + sum(lam_mg_d.*mg.P_lease_discharge) - sum(lam_buy.*mg.P_buy);
    g = L_dso + L_seso + L_mg;

    % AdaGrad update
    G_dso_c = G_dso_c + s_dso_c.^2;
    G_dso_d = G_dso_d + s_dso_d.^2;
    G_mg_c  = G_mg_c  + s_mg_c.^2;
    G_mg_d  = G_mg_d  + s_mg_d.^2;
    G_buy   = G_buy   + s_buy.^2;

    step_dso_c = alpha0 ./ (sqrt(G_dso_c) + eps0);
    step_dso_d = alpha0 ./ (sqrt(G_dso_d) + eps0);
    step_mg_c  = alpha0 ./ (sqrt(G_mg_c)  + eps0);
    step_mg_d  = alpha0 ./ (sqrt(G_mg_d)  + eps0);
    step_buy   = alpha0 ./ (sqrt(G_buy)   + eps0);

    lam_dso_c = lam_dso_c + step_dso_c .* s_dso_c;
    lam_dso_d = lam_dso_d + step_dso_d .* s_dso_d;
    lam_mg_c  = lam_mg_c  + step_mg_c  .* s_mg_c;
    lam_mg_d  = lam_mg_d  + step_mg_d  .* s_mg_d;
    lam_buy   = lam_buy   + step_buy   .* s_buy;

    % clip (helps numerics)
    lam_dso_c = min(max(lam_dso_c,-LAM_CLIP),LAM_CLIP);
    lam_dso_d = min(max(lam_dso_d,-LAM_CLIP),LAM_CLIP);
    lam_mg_c  = min(max(lam_mg_c,-LAM_CLIP),LAM_CLIP);
    lam_mg_d  = min(max(lam_mg_d,-LAM_CLIP),LAM_CLIP);
    lam_buy   = min(max(lam_buy,-LAM_CLIP),LAM_CLIP);

    % Ergodic averages (primal recovery)
    avg_dso_c = ((k-1)/k)*avg_dso_c + (1/k)*dso.P_charge;
    avg_dso_d = ((k-1)/k)*avg_dso_d + (1/k)*dso.P_discharge;
    avg_seso_c = ((k-1)/k)*avg_seso_c + (1/k)*seso.P_to_dso_charge;
    avg_seso_d = ((k-1)/k)*avg_seso_d + (1/k)*seso.P_to_dso_discharge;

    avg_mg_c = ((k-1)/k)*avg_mg_c + (1/k)*mg.P_lease_charge;
    avg_mg_d = ((k-1)/k)*avg_mg_d + (1/k)*mg.P_lease_discharge;
    avg_seso_mg_c = ((k-1)/k)*avg_seso_mg_c + (1/k)*seso.P_from_mg_charge;
    avg_seso_mg_d = ((k-1)/k)*avg_seso_mg_d + (1/k)*seso.P_from_mg_discharge;

    avg_buy_dso = ((k-1)/k)*avg_buy_dso + (1/k)*dso.P_to_mg;
    avg_buy_mg  = ((k-1)/k)*avg_buy_mg  + (1/k)*mg.P_buy;

    r_avg = max([ ...
        norm(avg_dso_c - avg_seso_c,2), ...
        norm(avg_dso_d - avg_seso_d,2), ...
        norm(avg_mg_c  - avg_seso_mg_c,2), ...
        norm(avg_mg_d  - avg_seso_mg_d,2), ...
        norm(avg_buy_dso - avg_buy_mg,2) ]);

    % record
    hist.r_pri_last(k)=r_last;
    hist.r_pri_avg(k)=r_avg;
    hist.g_dual(k)=g;
    if nargin>=2 && ~isempty(primal_ub) && primal_ub>0
        hist.dual_gap_rel(k)=max(0, (primal_ub - g)/abs(primal_ub));
    else
        hist.dual_gap_rel(k)=NaN;
    end
    hist.alpha_eff(k)=mean([mean(step_dso_c),mean(step_dso_d),mean(step_mg_c),mean(step_mg_d),mean(step_buy)]);
    hist.time(k)=toc(it);

    if mod(k,pe)==0 || k==1
        fprintf('[Dual-AdaGrad] k=%4d | r_last=%.3e r_avg=%.3e | g=%.3f | gap=%.3g | alpha_eff=%.3e | elapsed=%.1fs\n', ...
            k, r_last, r_avg, g, hist.dual_gap_rel(k), hist.alpha_eff(k), toc(t0));
    end

    % stopping based on ergodic primal residual (dual methods often use this)
    if r_avg <= par.alg.tol_pri
        hist = trim_hist(hist,k);
        break;
    end
    if toc(t0) > par.alg.max_walltime
        fprintf('[Dual-AdaGrad] Wall-time limit reached at k=%d.\n', k);
        hist = trim_hist(hist,k);
        break;
    end
end

out=struct();
out.method='DualDecomp-MirrorAdaGrad';
out.hist=hist;
out.final=struct('lambda',struct('dso_c',lam_dso_c,'dso_d',lam_dso_d,'mg_c',lam_mg_c,'mg_d',lam_mg_d,'buy',lam_buy));
out.comm.scalars_per_iter=5*T;
out.comm.total_scalars=numel(hist.r_pri_avg)*out.comm.scalars_per_iter;
end

function hist = trim_hist(hist,k)
fn = fieldnames(hist);
for i=1:numel(fn)
    v=hist.(fn{i});
    if isnumeric(v) && isvector(v) && numel(v)>=k
        hist.(fn{i}) = v(1:k);
    end
end
end
