function ops = toy_sdpsettings_v5(par)
ops = sdpsettings('verbose',par.solver.verbose,'solver',par.solver.name,'cachesolvers',1,'usex0',0);
try
    if strcmpi(par.solver.name,'gurobi')
        ops.gurobi.TimeLimit = par.solver.timelimit;
        ops.gurobi.OutputFlag = par.solver.verbose;
        ops.gurobi.Method = 1;
        ops.gurobi.DualReductions = 0;
    elseif strcmpi(par.solver.name,'mosek')
        ops.mosek.MSK_DPAR_OPTIMIZER_MAX_TIME = par.solver.timelimit;
    end
catch
end
end
