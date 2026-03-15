function out = Fun_DSO_Toy_v5(par, other_c, other_d, other_buy, lambda_c, lambda_d, lambda_buy, rho, mode)
T = par.T;

P_grid = sdpvar(1,T);
P_G    = sdpvar(1,T);
P_c    = sdpvar(1,T);
P_d    = sdpvar(1,T);
P_to_mg = sdpvar(1,T);

C=[];
C=[C, P_grid>=0, par.P_Gmin<=P_G<=par.P_Gmax];
C=[C, 0<=P_c<=par.P_dso_charge_max, 0<=P_d<=par.P_dso_discharge_max];
C=[C, 0<=P_to_mg<=par.P_mg_buy_max];
for tt=1:T
    C=[C, P_grid(tt)+P_G(tt)+P_d(tt)+par.P_R2(tt) == par.P_D2(tt)+par.P_D3(tt)+P_c(tt)+P_to_mg(tt)];
end

J = sum(par.c_grid.*P_grid + par.c_gen*P_G);

switch lower(mode)
    case {'admm','pen'}
        Obj = J ...
            + (rho/2)*sum((P_c-other_c).^2) + sum(lambda_c.*(P_c-other_c)) ...
            + (rho/2)*sum((P_d-other_d).^2) + sum(lambda_d.*(P_d-other_d)) ...
            + (rho/2)*sum((P_to_mg-other_buy).^2) + sum(lambda_buy.*(P_to_mg-other_buy));
    case 'dual'
        Obj = J + sum(lambda_c.*P_c) + sum(lambda_d.*P_d) + sum(lambda_buy.*P_to_mg);
    otherwise
        error('Unknown mode');
end

ops = toy_sdpsettings_v5(par);
sol = optimize(C, Obj, ops);

out=struct();
out.status=sol.problem;
out.P_charge=value(P_c);
out.P_discharge=value(P_d);
out.P_to_mg=value(P_to_mg);
out.J=value(J);
end
