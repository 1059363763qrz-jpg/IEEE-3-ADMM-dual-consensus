function out = Fun_SESO_Toy_v5(par, ...
    other_dso_c, other_dso_d, lambda_dso_c, lambda_dso_d, rho_dso, ...
    other_mg_c, other_mg_d, lambda_mg_c, lambda_mg_d, rho_mg, mode)

T = par.T;

P_s_ch  = sdpvar(1,T);
P_s_dis = sdpvar(1,T);
E_s     = sdpvar(1,T+1);

P_to_dso_c = sdpvar(1,T);
P_to_dso_d = sdpvar(1,T);

P_from_mg_c = sdpvar(1,T);
P_from_mg_d = sdpvar(1,T);

C=[];
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
if par.seso.Eend_eq_E0
    C=[C, E_s(T+1)==par.seso.E0];
end

J = sum(par.seso.c_ch*P_s_ch + par.seso.c_dis*P_s_dis + par.seso.c_lease*(P_from_mg_c+P_from_mg_d));

switch lower(mode)
    case {'admm','pen'}
        Obj = J ...
            + (rho_dso/2)*sum((P_to_dso_c-other_dso_c).^2) + sum(lambda_dso_c.*(P_to_dso_c-other_dso_c)) ...
            + (rho_dso/2)*sum((P_to_dso_d-other_dso_d).^2) + sum(lambda_dso_d.*(P_to_dso_d-other_dso_d)) ...
            + (rho_mg/2)*sum((P_from_mg_c-other_mg_c).^2) + sum(lambda_mg_c.*(P_from_mg_c-other_mg_c)) ...
            + (rho_mg/2)*sum((P_from_mg_d-other_mg_d).^2) + sum(lambda_mg_d.*(P_from_mg_d-other_mg_d));
    case 'dual'
        Obj = J ...
            + sum(lambda_dso_c.*P_to_dso_c) + sum(lambda_dso_d.*P_to_dso_d) ...
            + sum(lambda_mg_c.*P_from_mg_c) + sum(lambda_mg_d.*P_from_mg_d);
    otherwise
        error('Unknown mode');
end

ops = toy_sdpsettings_v5(par);
sol = optimize(C, Obj, ops);

out=struct();
out.status=sol.problem;
out.P_to_dso_charge=value(P_to_dso_c);
out.P_to_dso_discharge=value(P_to_dso_d);
out.P_from_mg_charge=value(P_from_mg_c);
out.P_from_mg_discharge=value(P_from_mg_d);
out.J=value(J);
end
