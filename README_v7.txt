ToyCase v7: Dual decomposition accelerated with Mirror Descent / AdaGrad (NO ADMM)

Run:
  addpath(genpath('toycase_v7'));
  toy_main_compare_v7_mirrorDual_v1

Key tuning parameters (in toy_build_params_v5.m or set in main):
  par.alg.alpha0        : base stepsize for dual mirror descent (default 0.05)
  par.alg.adagrad_eps   : epsilon for AdaGrad (default 1e-6)
  par.alg.lambda_clip   : clip bound for multipliers (default 200)
  par.alg.max_iter / par.alg.max_walltime / par.alg.tol_pri

Outputs:
  ToyCaseV7_Report.mat
  ToyCaseV7_Summary.csv
  Intermediate mats for each method.

Notes:
  - Dual(last) residual often does not converge for nonsmooth dual methods.
  - Dual(avg) (ergodic average residual) is the standard convergence indicator for primal feasibility under dual subgradient.
