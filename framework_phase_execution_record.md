# Bioinformatics Autoresearch Framework: Phase 1-5 详细执行记录

## 当前基线状态

截至当前仓库状态，已经完成的不是 full paper 级数据分析，而是一个 **task-aware prototype runtime**：

- `neoantigen`
- `hla_immunogenicity`
- `variant_prioritization`

这三个任务都已有 `smoke` 级入口，且以下 CLI 已可用：

- `python prepare.py --task <task_id> --mode smoke`
- `python train.py --task <task_id> --mode smoke --strategy <strategy> --run-id <id> --round-id <id>`
- `python confirm.py --task <task_id> --mode smoke --checkpoint <path>`
- `python blind_eval.py --task <task_id> --mode smoke --checkpoint <path>`
- `python controller.py run --task <task_id> --mode smoke --strategy <strategy> --run-id <id> --rounds <N>`

当前限制也非常明确：

- `full` 模式仍是硬阻断
- Task A / B / C 都还没有 full ingest
- Task A / B / C 都还没有正式 `source_manifest / data_card / lockbox_manifest`
- 论文级分析还不能开跑，必须先完成证据层冻结

下面记录的 Phase 1-5，是从当前原型状态推进到可投稿 framework paper 的正式路线。

---

## Phase 1: 冻结证据层

### 目标

把 framework 的“可审计性”从口头原则变成文件化、可检查、可复现的证据层。

这一阶段不追求训练结果，重点是：

- 数据来源可追溯
- split 规则可复现
- lockbox 边界不可混淆
- reviewer 能看懂每个任务到底纳入了什么、排除了什么

### 任务范围

本阶段只做：

- Task A: `neoantigen`
- Task B: `hla_immunogenicity`

Task C 不进入 Phase 1 主线。

### 需要创建的文件

建议统一放到：

- `manifests/neoantigen/`
- `manifests/hla_immunogenicity/`

每个任务至少包含：

- `data_card.md`
- `source_manifest.csv`
- `split_manifest.json`
- `lockbox_manifest.csv`
- `task_policy.md`

### 每个文件必须记录什么

#### `data_card.md`

必须写清楚：

- 任务定义
- 标签定义
- 正负样本来源
- 纳入标准
- 排除标准
- 人类 / HLA-I / peptide length 范围
- 是否包含 weak labels
- confirm / blind 的定义

#### `source_manifest.csv`

建议最少字段：

- `source_id`
- `source_name`
- `task_id`
- `source_type`
- `download_method`
- `license_or_access`
- `raw_file_path`
- `label_strength`
- `is_train_eligible`
- `is_confirm_eligible`
- `is_blind_only`
- `notes`

#### `split_manifest.json`

必须写清楚：

- exact dedup 规则
- mutation-event grouping 规则
- similarity grouping 规则
- study-aware grouping 规则
- confirm 划分规则
- blind 划分规则
- folds 数量
- 分层字段

#### `lockbox_manifest.csv`

建议最少字段：

- `sample_id`
- `task_id`
- `lockbox_name`
- `source_name`
- `reason`
- `allowed_for_training`
- `allowed_for_confirm`
- `allowed_for_blind`

#### `task_policy.md`

必须写清楚：

- 当前 task 的 hard constraints
- 不允许进入训练的来源
- 不允许 agent 接触的评估边界
- suspicious shortcut 列表
- baseline 优先级

### 具体步骤

1. 列出 Task A 当前打算纳入的所有公开数据源。
2. 列出 Task A 明确不进入训练的来源，尤其是 TESLA / 时间外推数据。
3. 为 Task A 写第一版 `source_manifest.csv`。
4. 为 Task A 写第一版 `data_card.md`。
5. 为 Task A 写第一版 `lockbox_manifest.csv`。
6. 为 Task A 写第一版 `split_manifest.json`。
7. 对 Task B 重复 1-6。
8. 审核两个任务的 label 定义是否一致到足够可比。
9. 审核 confirm / blind 路由是否会互相污染。
10. 把所有 manifest 引入 `prepare.py` 的 future full mode 设计说明里。

### 阶段产物

完成本阶段后，仓库里至少要有：

- `manifests/neoantigen/*`
- `manifests/hla_immunogenicity/*`
- 一个汇总性说明文件，说明 Task A / B 的 source 与 lockbox 边界

### 验收标准

- 第三方仅通过 manifest 文件，就能知道哪些数据能训练、哪些只能 confirm、哪些只能 blind
- 两个任务的 split 规则文字化且可重现
- 任何一个样本进入 blind 的原因都能被解释

### 停止条件

如果以下任一项不满足，不进入 Phase 2：

- Task A 的 lockbox 边界还说不清
- Task B 的 label 定义仍然含混
- split 规则还依赖“临时经验”而不是 manifest

---

## Phase 2: Task A Full Ingest

### 目标

让 `neoantigen` 从 smoke prototype 变成 full-ready task。

### 本阶段只解决什么

- Task A 原始数据拉取
- Task A 标准化表
- Task A full `prepare`
- Task A baseline leaderboard
- Task A `10-20` 轮 constrained sanity run

### 暂时不做什么

- Task B full ingest
- Task C full ingest
- 100 轮长轨迹

### 需要实现的能力

#### 数据层

- 原始源文件下载或本地挂载
- 原始快照写入 `data/raw/neoantigen/`
- 标准化表写入 `data/interim/neoantigen/`
- full processed dataset 写入 `data/processed/neoantigen/full/`

#### `prepare.py`

需要新增：

- `--task neoantigen --mode full`
- 读取 `manifests/neoantigen/source_manifest.csv`
- 读取 `manifests/neoantigen/lockbox_manifest.csv`
- 根据 manifest 决定 train / confirm / blind

#### 评估层

- full split manifest 落盘
- grouped CV 可跑
- confirm 可跑
- blind 可跑

### 具体步骤

1. 选定 Task A full ingest 的首批数据源。
2. 写数据下载或读取 adapter。
3. 实现 raw snapshot 保存。
4. 实现统一字段标准化。
5. 把 manifest 路由接入 `prepare.py`。
6. 在 full mode 下实现 exact dedup。
7. 在 full mode 下实现 grouped split。
8. 输出 full `dataset.parquet` 和 `split_manifest.json`。
9. 跑 baseline leaderboard。
10. 跑一次 `10-20` 轮 constrained sanity run。

### 阶段产物

- `data/processed/neoantigen/full/dataset.parquet`
- `data/processed/neoantigen/full/splits_grouped_v1.json`
- Task A baseline leaderboard
- Task A sanity report

### 验收标准

- `python prepare.py --task neoantigen --mode full` 可运行
- full baseline 可运行
- confirm / blind 可运行
- 至少有一次 controller sanity run 成功结束

### 停止条件

如果出现以下情况，不进入 Phase 3：

- full prepare 仍然需要手工拼接样本
- confirm / blind 还会污染
- baseline leaderboard 不能稳定复现

---

## Phase 3: Task B Full Ingest

### 目标

让 `hla_immunogenicity` 成为第二个真正可投稿的 full task。

### 本阶段解决什么

- Task B full 数据接入
- Task B full `prepare`
- Task B baseline leaderboard
- Task B `10-20` 轮 constrained sanity run

### 为什么这一步关键

Task B 不是为了多一个任务而多一个任务，而是为了证明：

- 这套 framework 不是只对 `neoantigen` 特化
- 同一套 evidence layer 哲学可以迁移到第二个近邻免疫任务

### 具体步骤

1. 明确 Task B 的数据源清单。
2. 明确 Task B 的正负标签定义。
3. 补 Task B 的 raw adapter。
4. 补 Task B 的 standardized schema。
5. 把 Task B manifest 路由接入 full `prepare.py`。
6. 产出 Task B full processed dataset。
7. 跑 Task B baseline leaderboard。
8. 跑 Task B `10-20` 轮 constrained sanity run。
9. 检查 confirm / blind 是否与 dev 趋势一致。

### 阶段产物

- `data/processed/hla_immunogenicity/full/dataset.parquet`
- `data/processed/hla_immunogenicity/full/splits_grouped_v1.json`
- Task B baseline leaderboard
- Task B sanity report

### 验收标准

- Task B full prepare 可运行
- Task B grouped CV 可运行
- Task B confirm / blind 可运行
- 至少 1 次 sanity run 完整结束

### 停止条件

如果 Task B 在下面任一项上失败，不进入 Phase 4：

- full dataset 仍然不稳定
- baseline 结果波动太大
- sanity run 明显出现 dev / blind 脱钩

---

## Phase 4: 主线 100 轮实验

### 目标

在 **Task A + Task B** 上形成 framework 论文的主 quantitative 证据。

### 本阶段固定实验矩阵

#### 主线

- Task A constrained: `3 runs x 100 rounds`
- Task B constrained: `3 runs x 100 rounds`

#### 对照

- Task A random: `3 runs x 20 rounds`
- Task B random: `3 runs x 20 rounds`
- Task A unconstrained: `3 runs x 20 rounds`
- Task B unconstrained: `3 runs x 20 rounds`

### 运行前必须先冻结什么

- Task A/B manifests 不再改
- full prepare 输出版本固定
- baseline leaderboard 固定
- `program.md` 版本固定
- `results.tsv` schema 固定

### 具体步骤

1. 固定主实验所用数据版本。
2. 固定主实验所用 `program.md`。
3. 清空并备份上一轮 `results.tsv`。
4. 先跑 Task A constrained `run 1-3`。
5. 汇总 Task A 每 run 的 best dev / confirm / blind。
6. 再跑 Task B constrained `run 1-3`。
7. 汇总 Task B 每 run 的 best dev / confirm / blind。
8. 再跑 Task A/B random 对照。
9. 再跑 Task A/B unconstrained 对照。
10. 汇总 accepted rounds、failed rounds、time-to-first-improvement、cost。

### 阶段产物

- Task A/B 主线结果矩阵
- Task A/B 对照结果矩阵
- 汇总版 `results.tsv`
- 每个 task / strategy / run 的 report

### 验收标准

- 六条 constrained 100 轮轨迹完整结束
- 对照轨迹完整结束
- 能输出均值、方差、提升频率、失败率
- 能形成 budget-matched comparison 表

### 停止条件

如果以下情况出现，需要先停下来分析，不继续盲跑更多轮：

- 100 轮内长期没有任何 keep
- confirm / blind 持续不跟 dev
- agent 只在低层 tweak 上打转
- worker failed 占比异常高

---

## Phase 5: Task C 与论文收口

### 目标

把 `variant_prioritization` 作为 framework 的跨域 strengthening case，并完成论文主文与补充材料。

### 本阶段的定位

Task C 是增强项，不应拖住主线投稿，但它能显著提升文章说服力。

### 推荐顺序

#### Step 1: 先做 Task C full readiness 判断

判断三件事：

- 是否已有足够稳定的数据源
- 是否能定义清晰的 grouped split
- 是否能设置独立 confirm / blind

#### Step 2: 如果可行，再做 Task C full ingest

只在以下前提下进入：

- Task A/B 主线已经跑完
- 论文主结果已经成型
- Task C 不会拖慢投稿节奏

#### Step 3: 论文收口

需要整理：

- framework overview 图
- Task summary 表
- budget-matched comparison 表
- transferability 图
- ablation 图
- discovered hypotheses 表
- failure taxonomy 表

### 论文主文至少要回答的 5 个问题

1. 为什么要冻结证据层？
2. 为什么 agent 只能在受限搜索面内行动？
3. 为什么 grouped CV / confirm / blind 会改变“什么叫进步”？
4. 为什么生物信息学里应该搜索高层归纳偏置，而不是只做低层调参？
5. 这套 framework 是否跨任务可迁移？

### 阶段产物

- Task C 结果或 Task C 可行性结论
- 主文图表初稿
- Supplementary 结果表
- Methods 草稿
- Results 草稿

### 验收标准

- 主文图表可以独立成立
- 所有主张都能在 results / manifests / reports 中找到证据
- 论文不依赖“口头解释”来补证据链

---

## 推荐执行顺序总结

按优先级，真正应该这样推进：

1. **Phase 1**  
   先冻结 Task A / B 的 manifests 和 evidence layer

2. **Phase 2**  
   先做 Task A full ingest，不要并行开 Task B

3. **Phase 3**  
   Task A 稳定后，再做 Task B full ingest

4. **Phase 4**  
   Task A / B 都 full-ready 后，再投入 100 轮主线和对照

5. **Phase 5**  
   Task C 作为增强项，服务于跨域性和论文收口

---

## 一句话原则

**先冻结证据，再接 full 数据；先做 Task A / B，再做 100 轮；先拿到可投稿主结果，再决定 Task C 做多深。**
