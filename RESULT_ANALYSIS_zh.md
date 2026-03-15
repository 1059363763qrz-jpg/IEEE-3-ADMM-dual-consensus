# ToyCase 最终结果分析（基于仓库已保存结果）

本文基于仓库中的两个汇总文件进行分析：

- `ToyCaseV7_Summary.csv`（v7，Dual 为全程平均版本）
- `ToyCaseV7_2_Summary.csv`（v7.2，Dual 为 suffix averaging 版本）

---

## 1) 先看 v7.2（当前推荐主脚本）

`ToyCaseV7_2_Summary.csv` 的关键数值如下：

- 集中式最优目标值：`central_obj = 71.9761`
- ADMM：184 次迭代，`r_pri=4.72e-3`，`r_dual=2.25e-2`，通信量 22080
- Dual(SuffixAvg)：500 次迭代（跑满上限），`r_last=6.42`，`r_suffixavg=1.19`，`dual_gap_rel=2.07e-3`，通信量 60000
- Penalty：82 次迭代，`r_pri=4.68e-3`，通信量 9840

### 结论 A：可行性收敛速度上，Penalty 和 ADMM 明显优于当前 Dual 配置

- ADMM 与 Penalty 的最终原始残差都在 `~4.7e-3`，已经达到 `tol_pri=5e-3` 的量级。
- Dual 的 `r_last` 与 `r_suffixavg` 仍显著偏大（尤其 `r_suffixavg=1.19`），说明在当前参数下 primal feasibility 恢复不足。

### 结论 B：Dual 的“对偶下界”是好的，但“原始可行恢复”仍是瓶颈

- `dual_gap_rel=2.07e-3`、`g_dual=71.8269` 与集中式值 71.9761 接近，说明下界质量不错；
- 但残差不够小，意味着“对偶接近”不等价于“原始一致性已恢复”。

### 结论 C：通信与时间成本上，Penalty 最省，ADMM 次之，Dual 最重

- 通信量：Penalty 9840 < ADMM 22080 << Dual 60000。
- 时间：ADMM 91.76s、Penalty 92.71s 基本同级；Dual 262.68s，约为 ADMM 的 2.86 倍。

---

## 2) v7 与 v7.2 的变化（看“后续改进是否有效”）

对比两个汇总文件可见：

1. **ADMM 明显更快**
   - 迭代从 368 降到 184（减半）
   - 总耗时从 1216.64s 降到 91.76s（显著下降）

2. **Dual 也更快但仍未达可行性目标**
   - 迭代从 1000 降到 500（减半）
   - 时间从 1304.40s 降到 262.68s
   - `r_last` 从 12.41 改善到 6.42（有改善），但仍远高于 ADMM/Penalty

3. **Penalty 基本稳定**
   - 两版均约 82 次迭代，残差与目标 gap 几乎一致

这说明：v7.2 在“效率”上整体优化明显，但 Dual 的核心短板仍是 primal residual。

---

## 3) 需要特别注意的数据异常

在 v7.2 汇总中：

- `admm_feas_obj = 0, admm_feas_gap_final = -1`
- `cons_feas_obj = 0, cons_feas_gap_final = -1`

这两个值是**异常值**（不是有效经济含义）。按主脚本意图，`*_feas_obj` 应该是将最终共识曲线回灌集中式后得到的“可行重评估目标值”，理论上应与 `central_obj` 同量纲、同数量级，而不是 0。

因此建议把这两列视为本次结果文件中的“失效字段”，本轮分析主要依据：

- 残差（`r_pri`, `r_dual`, `r_suffixavg`）
- 原始 gap（`admm_gap_final`, `cons_gap_final`）
- 对偶下界质量（`dual_gap_rel`, `g_dual`）
- 时间与通信

---

## 4) 面向实验报告的最终结论（可直接引用）

在当前仓库给定参数下：

1. **若目标是“快速获得可行一致解”**：Penalty 与 ADMM 都可用；Penalty 通信最低，ADMM 在目标值稳定性上更常被优先采用。  
2. **若目标是“对偶下界评估/理论验证”**：Dual(SuffixAvg) 仍有价值（下界接近集中式），但不能仅凭 `g_dual` 判断已得到高质量 primal 解。  
3. **若目标是“综合工程可用性（可行性 + 时间 + 通信）”**：本组结果里 Penalty/ADMM 明显优于当前 Dual 参数设定。  
4. **报告时需单独注明**：`*_feas_obj` 字段异常，不能用于最终结论，需要重新导出一次可行重评估结果再做严格最优性比较。

---

## 5) 下一步建议（最小改动优先）

1. 先修复/复核 `*_feas_obj` 导出流程，再复跑 v7.2。  
2. 对 Dual 做小范围网格：`alpha0`（0.01~0.08）、`burnin`（50/100/200）、`lambda_clip`（100/200/500），优先观察 `r_suffixavg`。  
3. 如需保证 primal recovery，可在 Dual 后增加一次“固定对偶或固定共识的投影/恢复”步骤，再评价可行目标值。

