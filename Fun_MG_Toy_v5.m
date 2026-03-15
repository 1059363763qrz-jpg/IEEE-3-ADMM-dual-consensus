function out = Fun_MG_Toy_v5(par, other_lease_c, other_lease_d, other_buy, lambda_c, lambda_d, lambda_buy, rho, mode)
T = par.T;

P_self_ch  = sdpvar(1,T);
P_self_dis = sdpvar(1,T);
P_lease_c  = sdpvar(1,T);
P_lease_d  = sdpvar(1,T);
P_buy      = sdpvar(1,T);
E_m        = sdpvar(1,T+1);

C=[];
C=[C, P_self_ch>=0, P_self_dis>=0, E_m(1)==par.mg.E0];
C=[C, 0<=P_lease_c<=par.P_mg_lease_charge_max, 0<=P_lease_d<=par.P_mg_lease_discharge_max];
C=[C, 0<=P_buy<=par.P_mg_buy_max];

for tt=1:T
    % MG balance with purchase
    C=[C, par.mg.P_R(tt)+P_self_dis(tt)+P_buy(tt) == par.mg.P_L(tt)+P_self_ch(tt)];
    % battery power (self+lease)
    C=[C, 0<=P_self_ch(tt)+P_lease_c(tt)<=par.mg.P_ch_max];
    C=[C, 0<=P_self_dis(tt)+P_lease_d(tt)<=par.mg.P_dis_max];
    % SOC update includes self+lease
    C=[C, E_m(tt+1)==E_m(tt)+par.mg.eta_ch*(P_self_ch(tt)+P_lease_c(tt)) ...
                  -(1/par.mg.eta_dis)*(P_self_dis(tt)+P_lease_d(tt))];
    C=[C, 0<=E_m(tt)<=par.mg.E_max];
end
C=[C, 0<=E_m(T+1)<=par.mg.E_max];
if par.mg.Eend_eq_E0
    C=[C, E_m(T+1)==par.mg.E0];
end

J = sum(par.mg.c_ch*(P_self_ch+P_lease_c) + par.mg.c_dis*(P_self_dis+P_lease_d) + par.c_grid.*P_buy);

switch lower(mode)
    case {'admm','pen'}
        Obj = J ...
            + (rho/2)*sum((P_lease_c-other_lease_c).^2) + sum(lambda_c.*(P_lease_c-other_lease_c)) ...
            + (rho/2)*sum((P_lease_d-other_lease_d).^2) + sum(lambda_d.*(P_lease_d-other_lease_d)) ...
            + (rho/2)*sum((P_buy-other_buy).^2) + sum(lambda_buy.*(P_buy-other_buy));
    case 'dual'
        Obj = J + sum(lambda_c.*P_lease_c) + sum(lambda_d.*P_lease_d) + sum(lambda_buy.*P_buy);
    otherwise
        error('Unknown mode');
end

ops = toy_sdpsettings_v5(par);
sol = optimize(C, Obj, ops);

out=struct();
out.status=sol.problem;
out.P_lease_charge=value(P_lease_c);
out.P_lease_discharge=value(P_lease_d);
out.P_buy=value(P_buy);
out.J=value(J);
end
