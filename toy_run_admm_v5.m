function out = toy_run_admm_v5(par)
T=par.T; K=par.alg.max_iter; rho=par.alg.rho; pe=par.alg.print_every;

% consensus
z_dso_c=zeros(1,T); z_dso_d=zeros(1,T);
z_mg_c=zeros(1,T);  z_mg_d=zeros(1,T);
z_buy=zeros(1,T);

% scaled duals
u_dso_c=zeros(1,T); u_dso_d=zeros(1,T); u_dso_buy=zeros(1,T);
u_seso_dso_c=zeros(1,T); u_seso_dso_d=zeros(1,T);
u_seso_mg_c=zeros(1,T);  u_seso_mg_d=zeros(1,T);
u_mg_c=zeros(1,T); u_mg_d=zeros(1,T); u_mg_buy=zeros(1,T);

hist.r_pri=zeros(K,1); hist.r_dual=zeros(K,1); hist.obj=zeros(K,1); hist.time=zeros(K,1);
t0=tic;

for k=1:K
    it=tic;
    dso_t=tic;
    dso = Fun_DSO_Toy_v5(par, z_dso_c, z_dso_d, z_buy, rho*u_dso_c, rho*u_dso_d, rho*u_dso_buy, rho, 'admm');
    td=toc(dso_t);

    seso_t=tic;
    seso = Fun_SESO_Toy_v5(par, z_dso_c, z_dso_d, rho*u_seso_dso_c, rho*u_seso_dso_d, rho, z_mg_c, z_mg_d, rho*u_seso_mg_c, rho*u_seso_mg_d, rho, 'admm');
    ts=toc(seso_t);

    mg_t=tic;
    mg = Fun_MG_Toy_v5(par, z_mg_c, z_mg_d, z_buy, rho*u_mg_c, rho*u_mg_d, rho*u_mg_buy, rho, 'admm');
    tm=toc(mg_t);

    if dso.status~=0 || seso.status~=0 || mg.status~=0
        warning('[ADMM] subproblem failed at k=%d (dso=%d seso=%d mg=%d)', k, dso.status, seso.status, mg.status);
        break;
    end

    z_dso_c_prev=z_dso_c; z_dso_d_prev=z_dso_d; z_mg_c_prev=z_mg_c; z_mg_d_prev=z_mg_d; z_buy_prev=z_buy;

    z_dso_c=0.5*(dso.P_charge+seso.P_to_dso_charge);
    z_dso_d=0.5*(dso.P_discharge+seso.P_to_dso_discharge);
    z_mg_c =0.5*(mg.P_lease_charge+seso.P_from_mg_charge);
    z_mg_d =0.5*(mg.P_lease_discharge+seso.P_from_mg_discharge);
    z_buy  =0.5*(dso.P_to_mg + mg.P_buy);

    u_dso_c = u_dso_c + (dso.P_charge - z_dso_c);
    u_dso_d = u_dso_d + (dso.P_discharge - z_dso_d);
    u_dso_buy = u_dso_buy + (dso.P_to_mg - z_buy);

    u_seso_dso_c = u_seso_dso_c + (seso.P_to_dso_charge - z_dso_c);
    u_seso_dso_d = u_seso_dso_d + (seso.P_to_dso_discharge - z_dso_d);

    u_mg_c = u_mg_c + (mg.P_lease_charge - z_mg_c);
    u_mg_d = u_mg_d + (mg.P_lease_discharge - z_mg_d);
    u_mg_buy = u_mg_buy + (mg.P_buy - z_buy);

    u_seso_mg_c = u_seso_mg_c + (seso.P_from_mg_charge - z_mg_c);
    u_seso_mg_d = u_seso_mg_d + (seso.P_from_mg_discharge - z_mg_d);

    r_pri = max([ ...
        norm(dso.P_charge - seso.P_to_dso_charge,2), ...
        norm(dso.P_discharge - seso.P_to_dso_discharge,2), ...
        norm(mg.P_lease_charge - seso.P_from_mg_charge,2), ...
        norm(mg.P_lease_discharge - seso.P_from_mg_discharge,2), ...
        norm(dso.P_to_mg - mg.P_buy,2) ]);

    r_dual = max([ ...
        rho*norm(z_dso_c-z_dso_c_prev,2), rho*norm(z_dso_d-z_dso_d_prev,2), ...
        rho*norm(z_mg_c-z_mg_c_prev,2),  rho*norm(z_mg_d-z_mg_d_prev,2), ...
        rho*norm(z_buy-z_buy_prev,2) ]);

    obj = dso.J + seso.J + mg.J;
    hist.r_pri(k)=r_pri; hist.r_dual(k)=r_dual; hist.obj(k)=obj; hist.time(k)=toc(it);

    if mod(k,pe)==0 || k==1
        fprintf('[ADMM] k=%3d | r_pri=%.3e r_dual=%.3e | obj=%.6f | solve(s) dso=%.2f seso=%.2f mg=%.2f | elapsed=%.1fs\n', ...
            k, r_pri, r_dual, obj, td, ts, tm, toc(t0));
    end

    if r_pri<=par.alg.tol_pri && r_dual<=par.alg.tol_dual
        hist=trim(hist,k); break;
    end
    if toc(t0)>par.alg.max_walltime
        fprintf('[ADMM] Wall-time limit reached at k=%d\n', k);
        hist=trim(hist,k); break;
    end
end

out=struct();
out.method='ADMM';
out.hist=hist;
out.final=struct('dso',dso,'seso',seso,'mg',mg, ...
    'z',struct('dso_c',z_dso_c,'dso_d',z_dso_d,'mg_c',z_mg_c,'mg_d',z_mg_d,'buy',z_buy));
out.comm.scalars_per_iter=5*T;
out.comm.total_scalars=numel(hist.obj)*out.comm.scalars_per_iter;
end

function hist=trim(hist,k)
hist.r_pri=hist.r_pri(1:k);
hist.r_dual=hist.r_dual(1:k);
hist.obj=hist.obj(1:k);
hist.time=hist.time(1:k);
end