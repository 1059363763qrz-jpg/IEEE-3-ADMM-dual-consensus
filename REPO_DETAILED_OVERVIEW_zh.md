# IEEE-3-ADMM-dual-consensus 代码详解（中文）

本文档从“建模对象—数学结构—算法实现—输出结果”四个层面梳理仓库实现，便于快速理解代码。

## 1. 这是在解什么问题？

该仓库实现了一个 **24 时段的三方协同调度凸优化 toy case**：

- DSO（配网侧）
- SESO（共享储能运营方）
- MG（微电网）

三方通过五类耦合功率序列达成一致（每个都是 24 维向量）：

1. DSO 充电请求 ↔ SESO 向 DSO 提供的充电能力
2. DSO 放电请求 ↔ SESO 向 DSO 提供的放电能力
3. MG 租赁充电能力 ↔ SESO 从 MG 接收的充电能力
4. MG 租赁放电能力 ↔ SESO 从 MG 接收的放电能力
5. DSO 对 MG 售电 ↔ MG 向 DSO 购电

核心是一个带储能 SOC 动态、功率上下限、末端 SOC 约束的一致性优化问题。

## 2. 参数与场景构造（`toy_build_params_v5.m`）

该文件构建了完整实验参数：

- 时间尺度：`T=24`
- 电价：分峰平谷 TOU 价 `c_grid`
- DSO 侧负荷与光伏剖面：`P_D2`, `P_D3`, `P_R2`
- DSO 可调机组：`P_Gmin`, `P_Gmax`, `c_gen`
- MG 侧负荷与光伏剖面：`mg.P_L`, `mg.P_R`
- SESO 与 MG 电池参数：`P_ch/dis_max`, `E_max`, `eta_ch/dis`, `E0`, `Eend_eq_E0`
- 交易/租赁上界：例如 `P_mg_buy_max`, `P_mg_lease_charge_max`
- 算法参数：`max_iter`, `tol_pri`, `tol_dual`, `rho`, `alpha0`, `burnin`, `beta_growth`
- 求解器参数：默认 `gurobi` + 每次子问题 `TimeLimit=10s`

一个关键建模设定是：**v5 允许 MG 从 DSO 购电**，从而提升整体可行性。

## 3. 集中式基准（`toy_solve_centralized_v5.m`）

该文件把三方变量放入一个统一优化中求解，作为最优基准 `obj*`：

- DSO 变量：`P_grid`, `P_G`, `P_dso_charge`, `P_dso_discharge`, `P_to_mg`
- SESO 变量：储能功率/能量 + 与 DSO/MG 的交互功率
- MG 变量：自有充放电、租赁充放电、购电 `P_m_buy`、SOC

约束层面包括：

- 三方各自的功率平衡 + 设备上下限
- 两个储能体（SESO/MG）的 SOC 递推与端点条件
- 五组一致性约束（例如 `P_to_dso_c == P_dso_charge`）

目标函数是三方成本加总：

- DSO：购电成本 + 机组成本
- SESO：充放电成本 + 租赁相关成本
- MG：充放电成本 + 购电成本

此外这个函数支持一个 `fixed` 可选输入，用于“固定交换功率剖面”，可把分布式算法产出的共识曲线带回集中式模型，计算其**全局可行重评估目标值**。

## 4. 三个局部子问题函数（`Fun_*.m`）

- `Fun_DSO_Toy_v5.m`
- `Fun_SESO_Toy_v5.m`
- `Fun_MG_Toy_v5.m`

三者结构一致：

1. 定义本地决策变量与本地物理约束
2. 定义本地成本 `J`
3. 按 `mode` 组装目标：
   - `admm` / `pen`：`J + 线性对偶项 + 二次惩罚项`
   - `dual`：`J + 线性拉格朗日项`（无增广项）
4. 调用 `optimize(...)` 得到本地最优响应

因此，主算法循环只是不断传入当前“其他方参考值”与“乘子/惩罚参数”，反复求这三个凸子问题。

## 5. 三类分布式算法实现

### 5.1 ADMM（`toy_run_admm_v5.m`）

实现要点：

- 维护共识变量 `z`（五组交换功率）
- 维护各参与方对应的 scaled dual `u`
- 每轮：
  1. DSO/SESO/MG 并行意义上的顺序求解（代码中顺序调用）
  2. 共识更新：`z = 0.5*(x_i + x_j)`
  3. 乘子更新：`u := u + (x-z)`
  4. 记录残差：
     - 原始残差 `r_pri`
     - 对偶残差 `r_dual`
- 停止条件：`r_pri <= tol_pri && r_dual <= tol_dual` 或超时

### 5.2 对偶分解 + Mirror/AdaGrad + 尾平均（`toy_run_dual_mirror_adagrad_suffixAvg_v1.m`）

这部分是仓库里最有研究意义的实现：

- 不用 ADMM 的增广项（`rho=0`），纯对偶分解
- 乘子更新采用 **逐坐标 AdaGrad 步长**：
  - 梯度累计：`G := G + s.^2`
  - 步长：`alpha0/(sqrt(G)+eps)` 并做 `clip`
- 乘子做盒约束裁剪（`lambda_clip`），防发散
- 记录两类可行性指标：
  - `r_pri_last`：末次迭代残差（常不稳定）
  - `r_pri_avg`：`burnin` 后尾平均残差（primal recovery）
- 记录 `g_dual`（对偶函数值下界）与 `dual_gap_rel`
- 用 `r_pri_avg` 作为主停机准则

这个实现体现了“**非增广对偶法要看平均点，而非只看最后一点**”的实验思想。

### 5.3 Penalty Consensus（`toy_run_penalty_consensus_v5.m`）

这是无显式对偶变量的惩罚一致性法：

- 每轮子问题仅含二次罚项（线性乘子全设 0）
- `z` 同样按双方均值更新
- 罚参数 `beta` 按 `beta_growth` 逐轮增大
- 只监控原始残差 `r_pri`

可以理解为“只靠罚项把一致性压出来”的基线。

## 6. 主控脚本（`toy_main_compare_v7_2_suffixDual_v1.m`）

该脚本按固定流程完成完整实验：

1. 构建参数
2. 求集中式基准（保存 `ToyCaseV7_2_Central.mat`）
3. 运行 ADMM（保存 `ToyCaseV7_2_ADMM.mat`）
4. 运行 Dual-Suffix（保存 `ToyCaseV7_2_Dual.mat`）
5. 运行 Penalty（保存 `ToyCaseV7_2_Consensus.mat`）
6. 汇总指标、画图、保存 `ToyCaseV7_2_Report.mat` 与 `ToyCaseV7_2_Summary.csv`

一个很实用的细节：

- ADMM / Penalty 的 `hist.obj` 可能来自不可行点，可能“假性优于”集中式；
- 脚本会把其最终 `z` 回灌到集中式模型做 **feasible re-eval**，得到 `admm_feas_obj` / `cons_feas_obj`，避免误判。

## 7. 输出指标如何理解

`summary` 里关键字段：

- `central_obj`: 集中式最优基准
- ADMM：`admm_r_end`, `admm_rd_end`, `admm_feas_gap_final`
- Dual：`dual_r_last_end`, `dual_r_avg_end`, `dual_gap_end`, `dual_g_end`
- Penalty：`cons_r_end`, `cons_feas_gap_final`
- `*_comm`: 按每轮 `5*T` 标量通信量累积

通常比较方式：

- 可行性：`r_pri` 或 Dual 的 `r_pri_avg`
- 近优性：与 `central_obj` 的 gap（优先看 feasible re-eval 后的 gap）
- 通信/时间：`*_comm` 与 `*_time`

## 8. 总结（这个仓库“实现了什么”）

一句话总结：

> 它实现了一个可复现实验平台：在同一三方储能协同调度模型上，系统比较 ADMM、纯对偶（Mirror/AdaGrad + 尾平均）和惩罚一致性三类分布式协调算法，并以集中式最优作为统一参照，输出可行性、最优性、通信和耗时等指标。

如果你接下来愿意，我可以继续给你做一版“数学形式化版”：把三方局部目标、五组耦合约束、拉格朗日函数、ADMM/对偶更新公式都写成统一符号推导版，直接可放论文附录。
