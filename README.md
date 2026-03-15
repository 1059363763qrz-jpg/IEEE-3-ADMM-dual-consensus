# IEEE-3-ADMM-dual-consensus

这个仓库是一个 **电力系统分布式优化 toy case（24 时段）**，用来比较 3 类分布式协调算法在同一问题上的表现：

- ADMM（带增广拉格朗日）
- 对偶分解 + Mirror Descent / AdaGrad（不使用 ADMM）
- 纯惩罚一致性（Penalty Consensus）

并用一个集中式优化解作为基准（ground truth）。

## 问题背景（模型对象）

系统包含三个参与方：

1. **DSO**（配电侧）
2. **SESO**（共享储能运营方）
3. **MG**（微电网）

三方在每个时段通过以下耦合变量进行协调：

- DSO 充/放电请求与 SESO 对 DSO 提供的充/放电能力
- MG 向 SESO 租赁的充/放电能力与 SESO 从 MG 收到的充/放电能力
- DSO 向 MG 售电与 MG 向 DSO 购电

本质上是一个多主体、带储能动态和能量平衡约束的凸优化分解问题。

## 代码结构

- `toy_build_params_v5.m`：构造负荷/PV/价格曲线、储能参数、算法参数与求解器参数。
- `toy_solve_centralized_v5.m`：集中式联合优化（作为最优目标值基准）。
- `Fun_DSO_Toy_v5.m` / `Fun_SESO_Toy_v5.m` / `Fun_MG_Toy_v5.m`：三方局部子问题。
- `toy_run_admm_v5.m`：ADMM 迭代。
- `toy_run_dual_mirror_adagrad_v1.m`：对偶镜像下降 + AdaGrad（全程平均）。
- `toy_run_dual_mirror_adagrad_suffixAvg_v1.m`：对偶镜像下降 + AdaGrad + 尾平均（suffix averaging）。
- `toy_run_penalty_consensus_v5.m`：惩罚一致性方法。
- `toy_main_compare_v7_mirrorDual_v1.m` / `toy_main_compare_v7_2_suffixDual_v1.m`：一键对比主脚本。

## 运行

推荐使用 v7.2 主脚本（尾平均版本）：

```matlab
addpath(genpath(pwd));
report = toy_main_compare_v7_2_suffixDual_v1;
```

会输出：

- `ToyCaseV7_2_Report.mat`
- `ToyCaseV7_2_Summary.csv`
- 中间算法结果 `.mat`

## 结果如何解读

- **ADMM / Penalty**：看原始残差（`r_pri`）和目标 gap。
- **Dual 类方法**：
  - `r_pri_last`（末次迭代残差）可能不单调、甚至不收敛。
  - `r_pri_avg`（平均/尾平均残差）更能反映 primal feasibility。
  - `g_dual` 是对偶下界，可结合集中式最优值看 `dual_gap_rel`。

## 依赖

- MATLAB
- YALMIP
- Gurobi（默认）或 Mosek（可切换）

## 说明

本仓库更偏向算法验证与对比，不是工业级配网仿真平台；但结构清晰，适合用于：

- 分布式优化算法教学演示
- ADMM 与非增广对偶法的行为对比
- 尾平均（suffix averaging）在对偶子梯度法中的效果验证
