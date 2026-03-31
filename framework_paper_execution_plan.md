# Bioinformatics Autoresearch Framework: 最小可发表实验清单

## 1. 论文定位

当前最稳的定位不是“发现了新的生物学规律”，而是：

**一种适用于高泄漏风险生物信息学任务的 constrained autoresearch framework。**

主张应收敛为三点：

1. 冻结证据层比扩大 agent 自由更重要。
2. 受限 agent 更适合搜索高层归纳偏置，而不是低层超参数。
3. 在固定预算下，这种框架比无约束搜索更稳定、更可审计、更容易迁移。

## 2. 最小可发表证据

如果要把文章写成 framework paper，最低证据配置建议如下：

| 维度 | 最小要求 |
|------|------|
| 任务数 | 2 |
| 每任务独立 run | 3 |
| 每 run 轮次 | 20 |
| 主要对照 | fixed baseline, random search, unconstrained agent |
| 必做消融 | grouped CV, confirm/blind separation, constrained search surface |
| 主要结果 | best dev, confirm, blind, accepted rounds, failure rate, cost |

这套矩阵的目标不是刷最高分，而是证明：

- framework 在多个任务上可迁移；
- 设计哲学会改变搜索行为；
- 提升不是偶然单次 hit。

## 3. 任务选择

### Task A: Neoantigen Ranking

这是当前主任务，也是论文的主案例。

保留理由：

- 已有完整原型和正向结果；
- 最适合讲“高泄漏风险 + grouped CV + blind isolation”；
- 已经跑出高层 WT-vs-Mut / HLA-conditioned contrast 改进。

当前状态：

- 现有 smoke 开发协议已可作为 prototype 证据；
- 后续应替换或扩展为真实规模数据集与更严格 lockbox。

### Task B: Human HLA-I Immunogenicity Ranking

这是最推荐补的第二任务。

推荐理由：

- 与当前系统共享 peptide/HLA 表征；
- 能测试 framework 的迁移性，而不需要彻底重写特征管线；
- 比直接跳到 variant pathogenicity 更节省工程成本。

选择原则：

- 人类样本优先；
- HLA-I 优先；
- 任务仍定义为 ranking；
- 保留 confirm / blind 路由。

### Task C: Optional Cross-Domain Task

如果要冲更强的 full paper，再补第三个任务：

- variant pathogenicity prioritization
- biomarker prioritization

这个任务不属于最小可发表集，但能显著增强“framework”主张。

## 4. 必做对照

### A. Fixed Baseline

每个任务都需要一个冻结 baseline leaderboard。

最低包含：

- logistic regression
- random forest
- xgboost
- lightgbm
- small MLP
- current fusion baseline

作用：

- 防止 agent 只是在打败一个过弱起点；
- 给 framework 提供固定参考线。

### B. Random Search

作用：

- 检查 autoresearch 是否只是“碰巧多试几次”。

实现要求：

- 与 constrained autoresearch 使用相同 edit budget；
- 每轮只允许在相同高层模块集合内随机采样改动；
- 同样使用 keep/revert 规则。

### C. Unconstrained Agent

作用：

- 证明限制搜索面是必要的，而不是多余的工程保守。

最低约束差异：

- constrained: 只改 `train.py`
- unconstrained: 允许 agent 同时触达训练脚本与部分配置层

注意：

- 不要让 unconstrained agent 触达真实 blind 标签；
- 这组实验的意义是“约束与不约束的差异”，不是安全性破坏演示。

### D. Optional Non-LLM Search

如果资源允许，建议补：

- Optuna
- Bayesian optimization

它们不是最小发表必要条件，但能更好回答 reviewer 的经典问题：

“这是否只是另一种更贵的超参数搜索？”

## 5. 必做消融

### Ablation 1: Grouped CV

比较：

- with grouped CV
- without grouped CV

目标：

- 证明更严格开发协议会改变“什么被认为是进步”。

预期结果：

- 去掉 grouped CV 后，dev 分数更容易虚高；
- confirm / blind 一致性变差；
- agent 更容易保留假进步。

### Ablation 2: Confirm / Blind Separation

比较：

- with confirm/blind isolation
- dev-only selection

目标：

- 证明独立 confirm / blind 路由是 framework 的必要组成。

预期结果：

- 没有 confirm/blind 时，更容易出现 leader overfit。

### Ablation 3: Search Surface Level

比较：

- high-level inductive bias search
- low-level hyperparameter / pooling / loss tweak search

目标：

- 证明 agent 在生物信息学里更应该搜索归纳偏置，而不是传统 AutoML 参数。

预期结果：

- 高层搜索更容易带来可迁移改动；
- 低层搜索更容易反复打平或局部过拟合。

## 6. 每个任务的运行矩阵

### 最小矩阵

| 组别 | runs | rounds/run | 说明 |
|------|------|------|------|
| baseline leaderboard | 1 | n/a | 冻结起点 |
| constrained autoresearch | 3 | 20 | 主结果 |
| random search | 3 | 20 | 预算公平对照 |
| unconstrained agent | 3 | 20 | 设计哲学对照 |

这意味着每个任务最少有 9 条搜索轨迹。

若做 2 个任务，则总共：

- 18 条搜索轨迹
- 每条 20 轮
- 共 360 轮

这是一个真实但仍可控的最小发表规模。

## 7. 记录哪些结果

每轮至少记录：

- task_id
- run_id
- round_id
- proposal summary
- hypothesis type
- changed module class
- dev score
- confirm score
- blind score
- keep / revert
- train failed / parse failed / timeout
- wall-clock time
- token usage
- lines changed

每条轨迹额外汇总：

- best dev score
- best confirm score
- best blind score
- time to first improvement
- accepted rounds
- failed rounds
- unique high-level hypotheses discovered

## 8. 论文图表清单

### 主文图

1. **Framework overview**
   展示 `prepare -> train -> confirm -> blind_eval -> controller -> keep/revert`。

2. **Search trajectory plot**
   每个任务画多条 run 的 round-by-round dev score 轨迹。

3. **Transferability summary**
   比较两个任务上 constrained autoresearch 与对照组的 best confirm / blind。

4. **Ablation figure**
   grouped CV、confirm/blind、high-level search 三个关键设计的影响。

### 主文表

1. **Task summary table**
   数据规模、label 类型、split 规则、blind 定义。

2. **Budget-matched comparison**
   constrained vs random vs unconstrained vs optional Optuna。

3. **Discovered hypothesis table**
   列出 agent 保留的高层改动类型，而不是只列 commit。

### Supplementary

1. 失败轮统计
2. 被 reject 的常见低收益方向
3. proposal schema 与 keep/revert 决策规则
4. per-length / per-HLA / per-study breakdown

## 9. 写进论文的方法学主线

论文不要主打“我们做出了最强 predictor”，而要主打：

1. 如何冻结证据层；
2. 如何限制 agent 搜索面；
3. 如何用可审计轨迹代替不可追溯试错；
4. 为什么生物信息学里真正值得搜索的是归纳偏置。

更具体地说：

- 主案例用 neoantigen 展开；
- 第二任务用来证明不是 task-specific；
- 消融用来证明 framework 的设计不是拍脑袋。

## 10. 当前仓库的 Phase 0

当前仓库已经可以支撑 Phase 0 原型证据：

- `python prepare.py --mode smoke`
- `python train.py --mode smoke --round-id 1`
- `python controller.py smoke --rounds 10`
- `python confirm.py --mode smoke --checkpoint <id>`
- `python blind_eval.py --mode smoke --checkpoint <id>`

这些命令主要用于：

- 证明系统闭环成立；
- 生成论文中的 prototype case study；
- 指导后续真实规模实验接口设计。

## 11. 下一阶段的实现顺序

### Phase 1: 固定第二任务

目标：

- 为 Task B 写清楚数据卡、split manifest、confirm/blind 规则；
- 尽量复用现有 peptide/HLA 特征层；
- 不要同时开启第三任务。

完成标志：

- 第二任务可以跑 baseline leaderboard；
- 第二任务有独立 confirm / blind；
- 第二任务能跑 1 次 smoke 级 controller。

### Phase 2: 跑最小矩阵

目标：

- 先跑 2 个任务 x 3 runs x 20 rounds 的 constrained autoresearch；
- 再补 random search 和 unconstrained agent。

完成标志：

- 所有轨迹结果落到统一的 `results.tsv` 或等价结构化日志；
- 能汇总 best/mean/std/failure rate/cost。

### Phase 3: 补消融

目标：

- grouped CV
- confirm/blind separation
- high-level vs low-level search

完成标志：

- 三个关键消融都能形成独立表格或图。

## 12. 停止标准

如果出现以下情况，应停止继续堆 agent 轮次，优先补任务或补证据：

1. 单任务上 round 数继续增加，但 confirm / blind 不再提升。
2. 多数 accepted round 只是微小低层改动，缺乏高层结构发现。
3. 第二任务迟迟无法复现任何正向收益。
4. dev 提升和 blind 提升严重脱钩。

## 13. 一句话结论

如果目标是发 **bioinformatics autoresearch framework**，那最小可发表路线不是“单任务跑更多轮”，而是：

**两个任务、固定预算、多次独立 run、关键消融、预算公平对照。**
