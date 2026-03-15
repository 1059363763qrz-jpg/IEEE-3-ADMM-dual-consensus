function out = toy_run_penalty_consensus_v5(par)
T=par.T; K=par.alg.max_iter; pe=par.alg.print_every;
beta=par.alg.beta0;

z_dso_c=zeros(1,T); z_dso_d=zeros(1,T);
z_mg_c=zeros(1,T);  z_mg_d=zeros(1,T);
z_buy=zeros(1,T);

hist.r_pri=zeros(K,1); hist.obj=zeros(K,1); hist.time=zeros(K,1); hist.beta=zeros(K,1);
t0=tic;

for k=1:K
    it=tic;

    dso_t=tic;
    dso = Fun_DSO_Toy_v5(par, z_dso_c, z_dso_d, z_buy, zeros(1,T), zeros(1,T), zeros(1,T), beta, 'pen'); td=toc(dso_t);

    seso_t=tic;
    seso = Fun_SESO_Toy_v5(par, z_dso_c, z_dso_d, zeros(1,T), zeros(1,T), beta, z_mg_c, z_mg_d, zeros(1,T), zeros(1,T), beta, 'pen'); ts=toc(seso_t);

    mg_t=tic;
    mg  = Fun_MG_Toy_v5(par, z_mg_c, z_mg_d, z_buy, zeros(1,T), zeros(1,T), zeros(1,T), beta, 'pen'); tm=toc(mg_t);

    if dso.status~=0 || seso.status~=0 || mg.status~=0
        warning('[Cons] subproblem failed at k=%d', k); break;
    end

    z_dso_c=0.5*(dso.P_charge+seso.P_to_dso_charge);
    z_dso_d=0.5*(dso.P_discharge+seso.P_to_dso_discharge);
    z_mg_c =0.5*(mg.P_lease_charge+seso.P_from_mg_charge);
    z_mg_d =0.5*(mg.P_lease_discharge+seso.P_from_mg_discharge);
    z_buy  =0.5*(dso.P_to_mg + mg.P_buy);

    r_pri = max([ ...
        norm(dso.P_charge - seso.P_to_dso_charge,2), ...
        norm(dso.P_discharge - seso.P_to_dso_discharge,2), ...
        norm(mg.P_lease_charge - seso.P_from_mg_charge,2), ...
        norm(mg.P_lease_discharge - seso.P_from_mg_discharge,2), ...
        norm(dso.P_to_mg - mg.P_buy,2) ]);

    obj = dso.J + seso.J + mg.J;
    hist.r_pri(k)=r_pri; hist.obj(k)=obj; hist.time(k)=toc(it); hist.beta(k)=beta;

    if mod(k,pe)==0 || k==1
        fprintf('[Cons] k=%3d | r_pri=%.3e | obj=%.6f | beta=%.2e | solve(s) dso=%.2f seso=%.2f mg=%.2f | elapsed=%.1fs\n', ...
            k, r_pri, obj, beta, td, ts, tm, toc(t0));
    end

    if r_pri<=par.alg.tol_pri
        hist=trim(hist,k); break;
    end

    beta = beta*par.alg.beta_growth;

    if toc(t0)>par.alg.max_walltime
        fprintf('[Cons] Wall-time limit reached at k=%d\n', k);
        hist=trim(hist,k); break;
    end
end

out=struct();
out.method='PenaltyConsensus';
out.hist=hist;
out.final=struct('dso',dso,'seso',seso,'mg',mg,'beta',beta, ...
    'z',struct('dso_c',z_dso_c,'dso_d',z_dso_d,'mg_c',z_mg_c,'mg_d',z_mg_d,'buy',z_buy));
out.comm.scalars_per_iter=5*T;
out.comm.total_scalars=numel(hist.obj)*out.comm.scalars_per_iter;
end

function hist=trim(hist,k)
hist.r_pri=hist.r_pri(1:k);
hist.obj=hist.obj(1:k);
hist.time=hist.time(1:k);
hist.beta=hist.beta(1:k);
end