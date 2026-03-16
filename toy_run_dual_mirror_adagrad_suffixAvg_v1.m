function out = toy_run_dual_mirror_adagrad_suffixAvg_v1(par, primal_ub)
% Dual decomposition with Mirror Descent / AdaGrad diagonal scaling + suffix averaging.
% NO augmented Lagrangian (NO ADMM).
%
% Outputs:
%   hist.r_pri_last : last-iterate residual (may not converge)
%   hist.r_pri_avg  : suffix-averaged residual after burn-in (primal recovery)
%   hist.g_dual     : dual function value (lower bound on primal optimum)
%   hist.dual_gap_rel : (p* - g)/p*
%
% Parameters (in par.alg, with defaults):
%   alpha0        : base stepsize (default 0.05)
%   adagrad_eps   : eps in AdaGrad denom (default 1e-3)
%   step_clip     : max element-wise stepsize (default 5.0)
%   burnin        : start averaging after this iter (default 100)
%   lambda_clip   : clip bound for multipliers (default 200)

T = par.T;
K = par.alg.max_iter;
pe = par.alg.print_every;

% multipliers
lam_dso_c = zeros(1,T); lam_dso_d = zeros(1,T);
lam_mg_c  = zeros(1,T); lam_mg_d  = zeros(1,T);
lam_buy   = zeros(1,T);

% AdaGrad accumulators
G_dso_c = zeros(1,T); G_dso_d = zeros(1,T);
G_mg_c  = zeros(1,T); G_mg_d  = zeros(1,T);
G_buy   = zeros(1,T);

% hyperparameters
alpha0 = 0.05;
if isfield(par.alg,'alpha0'); alpha0 = par.alg.alpha0; end
eps0 = 1e-3;
if isfield(par.alg,'adagrad_eps'); eps0 = par.alg.adagrad_eps; end
step_clip = 5.0;
if isfield(par.alg,'step_clip'); step_clip = par.alg.step_clip; end
burnin = 100;
if isfield(par.alg,'burnin'); burnin = par.alg.burnin; end
LAM_CLIP = 200;
if isfield(par.alg,'lambda_clip'); LAM_CLIP = par.alg.lambda_clip; end
mom = 0.5;
if isfield(par.alg,'dual_momentum'); mom = par.alg.dual_momentum; end
restart_ratio = 1.05;
if isfield(par.alg,'dual_restart_ratio'); restart_ratio = par.alg.dual_restart_ratio; end
sclip = 2.0;
if isfield(par.alg,'dual_subgrad_clip'); sclip = par.alg.dual_subgrad_clip; end
weighted_suffix = true;
if isfield(par.alg,'dual_suffix_weighted'); weighted_suffix = par.alg.dual_suffix_weighted; end
polyak_blend = 0.35;
if isfield(par.alg,'dual_polyak_blend'); polyak_blend = par.alg.dual_polyak_blend; end
decay_power = 0.25;
if isfield(par.alg,'dual_decay_power'); decay_power = par.alg.dual_decay_power; end
decay_warmup = 50;
if isfield(par.alg,'dual_decay_warmup'); decay_warmup = par.alg.dual_decay_warmup; end
auto_shrink = true;
if isfield(par.alg,'dual_auto_shrink'); auto_shrink = par.alg.dual_auto_shrink; end
patience = 30;
if isfield(par.alg,'dual_patience'); patience = par.alg.dual_patience; end
shrink = 0.7;
if isfield(par.alg,'dual_shrink'); shrink = par.alg.dual_shrink; end
min_scale = 0.05;
if isfield(par.alg,'dual_min_scale'); min_scale = par.alg.dual_min_scale; end

% momentum memory
lam_prev_dso_c = lam_dso_c; lam_prev_dso_d = lam_dso_d;
lam_prev_mg_c  = lam_mg_c;  lam_prev_mg_d  = lam_mg_d;
lam_prev_buy   = lam_buy;
r_prev = inf;

% suffix averages (start after burnin)
kavg = 0;
avg_dso_c=zeros(1,T); avg_dso_d=zeros(1,T);
avg_seso_c=zeros(1,T); avg_seso_d=zeros(1,T);
avg_mg_c=zeros(1,T);  avg_mg_d=zeros(1,T);
avg_seso_mg_c=zeros(1,T); avg_seso_mg_d=zeros(1,T);
avg_buy_dso=zeros(1,T); avg_buy_mg=zeros(1,T);

hist.r_pri_last=zeros(K,1);
hist.r_pri_avg=nan(K,1);
hist.g_dual=zeros(K,1);
hist.dual_gap_rel=nan(K,1);
hist.step_mean=zeros(K,1);
hist.step_max=zeros(K,1);
hist.step_scale=zeros(K,1);
hist.time=zeros(K,1);

step_scale = 1.0;
best_r_avg = inf;
stale_cnt = 0;

t0=tic;
for k=1:K
    it=tic;

    % Solve subproblems (dual mode)
    dso  = Fun_DSO_Toy_v5(par, [], [], [], lam_dso_c, lam_dso_d, lam_buy, 0, 'dual');
    seso = Fun_SESO_Toy_v5(par, [], [], -lam_dso_c, -lam_dso_d, 0, [], [], -lam_mg_c, -lam_mg_d, 0, 'dual');
    mg   = Fun_MG_Toy_v5(par, [], [], [], +lam_mg_c, +lam_mg_d, -lam_buy, 0, 'dual');

    if dso.status~=0 || seso.status~=0 || mg.status~=0
        warning('[Dual-AdaGrad-Suffix] subproblem failed at k=%d (dso=%d seso=%d mg=%d).', k, dso.status, seso.status, mg.status);
        break;
    end

    % residuals (subgradients)
    s_dso_c = dso.P_charge    - seso.P_to_dso_charge;
    s_dso_d = dso.P_discharge - seso.P_to_dso_discharge;
    s_mg_c  = mg.P_lease_charge    - seso.P_from_mg_charge;
    s_mg_d  = mg.P_lease_discharge - seso.P_from_mg_discharge;
    s_buy   = dso.P_to_mg - mg.P_buy;

    % subgradient clipping (element-wise)
    s_dso_c = min(max(s_dso_c,-sclip),sclip);
    s_dso_d = min(max(s_dso_d,-sclip),sclip);
    s_mg_c  = min(max(s_mg_c,-sclip),sclip);
    s_mg_d  = min(max(s_mg_d,-sclip),sclip);
    s_buy   = min(max(s_buy,-sclip),sclip);

    r_last = max([norm(s_dso_c,2), norm(s_dso_d,2), norm(s_mg_c,2), norm(s_mg_d,2), norm(s_buy,2)]);

    % dual function value g(lambda)
    L_dso  = dso.J  + sum(lam_dso_c.*dso.P_charge) + sum(lam_dso_d.*dso.P_discharge) + sum(lam_buy.*dso.P_to_mg);
    L_seso = seso.J - sum(lam_dso_c.*seso.P_to_dso_charge) - sum(lam_dso_d.*seso.P_to_dso_discharge) ...
                    - sum(lam_mg_c.*seso.P_from_mg_charge) - sum(lam_mg_d.*seso.P_from_mg_discharge);
    L_mg   = mg.J   + sum(lam_mg_c.*mg.P_lease_charge) + sum(lam_mg_d.*mg.P_lease_discharge) - sum(lam_buy.*mg.P_buy);
    g = L_dso + L_seso + L_mg;

    % AdaGrad accumulators
    G_dso_c = G_dso_c + s_dso_c.^2;
    G_dso_d = G_dso_d + s_dso_d.^2;
    G_mg_c  = G_mg_c  + s_mg_c.^2;
    G_mg_d  = G_mg_d  + s_mg_d.^2;
    G_buy   = G_buy   + s_buy.^2;

    % element-wise stepsizes + clip + global decay
    if k <= decay_warmup
        decay_k = 1.0;
    else
        decay_k = ((k-decay_warmup)+1)^(-decay_power);
    end
    step_dso_c = min(step_clip, step_scale * decay_k * alpha0 ./ (sqrt(G_dso_c) + eps0));
    step_dso_d = min(step_clip, step_scale * decay_k * alpha0 ./ (sqrt(G_dso_d) + eps0));
    step_mg_c  = min(step_clip, step_scale * decay_k * alpha0 ./ (sqrt(G_mg_c)  + eps0));
    step_mg_d  = min(step_clip, step_scale * decay_k * alpha0 ./ (sqrt(G_mg_d)  + eps0));
    step_buy   = min(step_clip, step_scale * decay_k * alpha0 ./ (sqrt(G_buy)   + eps0));

    % Optional Polyak-style scalar step blended with AdaGrad (still pure dual).
    if ~isempty(primal_ub) && primal_ub>0 && polyak_blend>0
        denom = sum(s_dso_c.^2) + sum(s_dso_d.^2) + sum(s_mg_c.^2) + sum(s_mg_d.^2) + sum(s_buy.^2);
        alpha_poly = max(0, (primal_ub - g) / max(1e-9, denom));
        alpha_poly = min(step_clip, step_scale*decay_k*alpha_poly);
        step_dso_c = (1-polyak_blend)*step_dso_c + polyak_blend*alpha_poly;
        step_dso_d = (1-polyak_blend)*step_dso_d + polyak_blend*alpha_poly;
        step_mg_c  = (1-polyak_blend)*step_mg_c  + polyak_blend*alpha_poly;
        step_mg_d  = (1-polyak_blend)*step_mg_d  + polyak_blend*alpha_poly;
        step_buy   = (1-polyak_blend)*step_buy   + polyak_blend*alpha_poly;
    end

    % Optional Polyak-style scalar step blended with AdaGrad (still pure dual).
    if ~isempty(primal_ub) && primal_ub>0 && polyak_blend>0
        denom = sum(s_dso_c.^2) + sum(s_dso_d.^2) + sum(s_mg_c.^2) + sum(s_mg_d.^2) + sum(s_buy.^2);
        alpha_poly = max(0, (primal_ub - g) / max(1e-9, denom));
        alpha_poly = min(step_clip, alpha_poly);
        step_dso_c = (1-polyak_blend)*step_dso_c + polyak_blend*alpha_poly;
        step_dso_d = (1-polyak_blend)*step_dso_d + polyak_blend*alpha_poly;
        step_mg_c  = (1-polyak_blend)*step_mg_c  + polyak_blend*alpha_poly;
        step_mg_d  = (1-polyak_blend)*step_mg_d  + polyak_blend*alpha_poly;
        step_buy   = (1-polyak_blend)*step_buy   + polyak_blend*alpha_poly;
    end

    % update multipliers
    mom_eff = mom;
    if k>1 && r_last > restart_ratio*r_prev
        mom_eff = 0; % adaptive restart when primal residual worsens
    end
    lam_new_dso_c = lam_dso_c + step_dso_c .* s_dso_c + mom_eff*(lam_dso_c - lam_prev_dso_c);
    lam_new_dso_d = lam_dso_d + step_dso_d .* s_dso_d + mom_eff*(lam_dso_d - lam_prev_dso_d);
    lam_new_mg_c  = lam_mg_c  + step_mg_c  .* s_mg_c  + mom_eff*(lam_mg_c  - lam_prev_mg_c);
    lam_new_mg_d  = lam_mg_d  + step_mg_d  .* s_mg_d  + mom_eff*(lam_mg_d  - lam_prev_mg_d);
    lam_new_buy   = lam_buy   + step_buy   .* s_buy   + mom_eff*(lam_buy   - lam_prev_buy);

    lam_prev_dso_c = lam_dso_c; lam_prev_dso_d = lam_dso_d;
    lam_prev_mg_c  = lam_mg_c;  lam_prev_mg_d  = lam_mg_d;
    lam_prev_buy   = lam_buy;
    lam_dso_c = lam_new_dso_c; lam_dso_d = lam_new_dso_d;
    lam_mg_c  = lam_new_mg_c;  lam_mg_d  = lam_new_mg_d;
    lam_buy   = lam_new_buy;

    % clip multipliers
    lam_dso_c = min(max(lam_dso_c,-LAM_CLIP),LAM_CLIP);
    lam_dso_d = min(max(lam_dso_d,-LAM_CLIP),LAM_CLIP);
    lam_mg_c  = min(max(lam_mg_c,-LAM_CLIP),LAM_CLIP);
    lam_mg_d  = min(max(lam_mg_d,-LAM_CLIP),LAM_CLIP);
    lam_buy   = min(max(lam_buy,-LAM_CLIP),LAM_CLIP);

    % suffix averaging after burnin
    if k >= burnin
        if weighted_suffix
            w_new = k - burnin + 1;
            w = w_new/(kavg + w_new);
            kavg = kavg + w_new;
        else
            kavg = kavg + 1;
            w = 1/kavg; % incremental average
        end
        avg_dso_c = (1-w)*avg_dso_c + w*dso.P_charge;
        avg_dso_d = (1-w)*avg_dso_d + w*dso.P_discharge;
        avg_seso_c = (1-w)*avg_seso_c + w*seso.P_to_dso_charge;
        avg_seso_d = (1-w)*avg_seso_d + w*seso.P_to_dso_discharge;

        avg_mg_c = (1-w)*avg_mg_c + w*mg.P_lease_charge;
        avg_mg_d = (1-w)*avg_mg_d + w*mg.P_lease_discharge;
        avg_seso_mg_c = (1-w)*avg_seso_mg_c + w*seso.P_from_mg_charge;
        avg_seso_mg_d = (1-w)*avg_seso_mg_d + w*seso.P_from_mg_discharge;

        avg_buy_dso = (1-w)*avg_buy_dso + w*dso.P_to_mg;
        avg_buy_mg  = (1-w)*avg_buy_mg  + w*mg.P_buy;

        r_avg = max([ ...
            norm(avg_dso_c - avg_seso_c,2), ...
            norm(avg_dso_d - avg_seso_d,2), ...
            norm(avg_mg_c  - avg_seso_mg_c,2), ...
            norm(avg_mg_d  - avg_seso_mg_d,2), ...
            norm(avg_buy_dso - avg_buy_mg,2) ]);
        hist.r_pri_avg(k) = r_avg;

        if auto_shrink
            if r_avg < (1-1e-3)*best_r_avg
                best_r_avg = r_avg;
                stale_cnt = 0;
            else
                stale_cnt = stale_cnt + 1;
                if stale_cnt >= patience
                    step_scale = max(min_scale, step_scale * shrink);
                    stale_cnt = 0;
                end
            end
        end
    end

    r_prev = r_last;

    % record
    hist.r_pri_last(k)=r_last;
    hist.g_dual(k)=g;
    if ~isempty(primal_ub) && primal_ub>0
        hist.dual_gap_rel(k)=max(0, (primal_ub - g)/abs(primal_ub));
    end
    all_steps = [step_dso_c(:);step_dso_d(:);step_mg_c(:);step_mg_d(:);step_buy(:)];
    hist.step_mean(k)=mean(all_steps);
    hist.step_max(k)=max(all_steps);
    hist.step_scale(k)=step_scale;
    hist.time(k)=toc(it);

    if mod(k,pe)==0 || k==1
        if isnan(hist.r_pri_avg(k))
            fprintf('[Dual-AdaGrad-Suffix] k=%4d | r_last=%.3e | g=%.3f | gap=%.3g | step_mean=%.3e step_max=%.3e scale=%.2f | elapsed=%.1fs\n', ...
                k, r_last, g, hist.dual_gap_rel(k), hist.step_mean(k), hist.step_max(k), hist.step_scale(k), toc(t0));
        else
            fprintf('[Dual-AdaGrad-Suffix] k=%4d | r_last=%.3e r_avg=%.3e | g=%.3f | gap=%.3g | step_mean=%.3e step_max=%.3e scale=%.2f | elapsed=%.1fs\n', ...
                k, r_last, hist.r_pri_avg(k), g, hist.dual_gap_rel(k), hist.step_mean(k), hist.step_max(k), hist.step_scale(k), toc(t0));
        end
    end

    % stopping: use suffix-avg residual
    if k >= burnin && hist.r_pri_avg(k) <= par.alg.tol_pri
        hist = trim_hist(hist,k);
        break;
    end
    if toc(t0) > par.alg.max_walltime
        fprintf('[Dual-AdaGrad-Suffix] Wall-time limit reached at k=%d.\n', k);
        hist = trim_hist(hist,k);
        break;
    end
end

out=struct();
out.method='DualDecomp-MirrorAdaGrad-Suffix';
out.hist=hist;
out.final=struct('lambda',struct('dso_c',lam_dso_c,'dso_d',lam_dso_d,'mg_c',lam_mg_c,'mg_d',lam_mg_d,'buy',lam_buy), ...
                 'burnin',burnin,'kavg',kavg);
out.comm.scalars_per_iter=5*T;
out.comm.total_scalars=numel(hist.r_pri_last)*out.comm.scalars_per_iter;
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
