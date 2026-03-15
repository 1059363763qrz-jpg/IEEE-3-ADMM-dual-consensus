ToyCase v7.2: Dual decomposition accelerated with Mirror Descent / AdaGrad + Suffix Averaging (NO ADMM)

Run:
  addpath(genpath('toycase_v7_2'));
  toy_main_compare_v7_2_suffixDual_v1

Key tuning parameters (edit toy_build_params_v5.m or set in main):
  par.alg.alpha0      : base stepsize (default 0.05)
  par.alg.adagrad_eps : AdaGrad epsilon (default 1e-3)
  par.alg.step_clip   : stepsize cap (default 5.0)
  par.alg.burnin      : burn-in iterations for suffix averaging (default 100)
  par.alg.lambda_clip : multiplier clip (default 200)
  par.alg.max_iter / max_walltime / tol_pri

Outputs:
  ToyCaseV7_2_Report.mat
  ToyCaseV7_2_Summary.csv
  plus intermediate mats for each method.
