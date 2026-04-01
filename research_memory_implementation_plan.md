# AutoNeoAg 研究记忆机制实施方案

更新时间：2026-04-01  
适用范围：`controller.py`、`src/autoneoag/runtime/results.py`、`src/autoneoag/runtime/codex_worker.py`、`schemas/codex_worker_output.schema.json`

## 1. 背景与目标

当前 AutoNeoAg 的 `constrained` 搜索已经不再是完全随机试错。根据 [results.tsv](/Volumes/KINGSTON/work/research/AutoNeoAg/results.tsv) 的现有轨迹，`neoantigen` 全量运行已经表现出明确的方向性：

- `round 17` 与 `round 19` 都来自 `HLA-conditioned preference / competition` 这条线
- `gating`、`ranking_objective`、以及若干“直接向已有 head 硬塞更多 context”的方案连续退化
- 从 `round 20` 开始，大量尝试都属于成功方向的近邻变体，但没有继续超过当前 champion

这说明下一版 agent 的核心问题不是“知识不够”，而是“没有显式研究记忆”。  
本方案的目标是把当前 blind search 升级为一套**可审计、可压缩、可利用历史结果的方向性搜索机制**。

目标行为：

- agent 能从历史结果中判断哪些方向有效
- agent 优先沿有效方向做小步优化
- controller 对重复失败方向做降权或临时冻结
- prompt 不再堆叠长历史，而是提供结构化 frontier hint
- 每轮搜索都留下可回溯的“方向选择依据”
- family 标签不只依赖 worker 自报，而是保留 controller 推断结果
- family 评价不只围绕 dev，也允许低频 confirm 回灌

一句话定义：把当前搜索流程从 `blind search` 升级为 `frontier-guided constrained autoresearch`。

## 2. 当前实现现状

### 2.1 当前 `results.tsv`

当前 [results.tsv](/Volumes/KINGSTON/work/research/AutoNeoAg/results.tsv) 只记录以下字段：

- `task_id`
- `strategy`
- `run_id`
- `round_id`
- `commit`
- `dev_score`
- `confirm_score`
- `blind_score`
- `status`
- `failure_type`
- `training_seconds`
- `lines_changed`
- `description`

它能记录实验轨迹，但**不能表达方向家族、父代关系、搜索模式、连续失败模式**，因此 controller 无法自动形成“研究记忆”。

### 2.2 当前 controller 行为

[controller.py](/Volumes/KINGSTON/work/research/AutoNeoAg/controller.py) 当前每轮只做这几件事：

1. 调用 worker 生成一个小改动
2. 训练并读取 `val_score`
3. `keep` 或 `discard`
4. 把 proposal JSON 追加到 `summary_lines`
5. 下一轮把最近几条 `summary_lines` 原样塞回 worker

问题在于：

- 最近摘要只是“原始历史”，不是“结构化方向提示”
- controller 不知道哪些 family 已经连续失败
- controller 不知道当前 champion 属于哪条家族
- controller 不会做 exploit / explore / avoid 调度

### 2.3 当前 worker 输出

[schemas/codex_worker_output.schema.json](/Volumes/KINGSTON/work/research/AutoNeoAg/schemas/codex_worker_output.schema.json) 当前只要求：

- `hypothesis`
- `expected_change`
- `risk`
- `edit_scope`
- `summary`

这意味着 worker 输出对人可读，但对 controller 来说可分析性不足，无法稳定归档到“方向家族”。

## 3. 设计原则

本机制采用以下四条原则：

### 3.1 证据层保持冻结

本方案**不改变**下面这些机制的权责边界：

- `prepare.py`
- `confirm.py`
- `blind_eval.py`
- keep/revert 由确定性 controller 决定

研究记忆机制只影响：

- proposal 生成前的上下文压缩
- proposal 的家族标注
- controller 对家族的调度策略

### 3.2 历史不是原样堆叠，而是压缩成 frontier

agent 不应该阅读全部历史，而应该阅读 controller 生成的**frontier state**：

- 当前 champion 是谁
- 哪些 family 最近有效
- 哪些 family 最近在退化
- 哪些 pattern 不允许继续重复
- 本轮更适合 exploit 还是 explore

### 3.3 搜索不是平均分配，而是 exploit 优先

默认调度策略：

- `70%` 轮次：`exploit`
- `20%` 轮次：`explore`
- `10%` 轮次：`recovery`

其中：

- `exploit`：沿当前 champion family 做一阶邻域搜索
- `explore`：尝试相邻 family，但保持 edit 半径小
- `recovery`：当连续多轮无新 best 时，切到次优 family 或新 family

### 3.4 方向家族必须可审计

controller 的判断不能只存在 prompt 里，必须落成文件：

- `frontier_state.json`
- `frontier_hint.md`
- `family_stats.tsv`

这样论文写作和后期失败分析才有依据。

### 3.5 family 标签不能完全依赖 worker 自报

worker 输出的 family 很有用，但不能当作真值。  
更稳的机制是同时保留：

- `worker_declared_family`
- `controller_inferred_family`
- `proposal_family`

语义分别是：

- `worker_declared_family`：worker 自报的方向标签
- `controller_inferred_family`：controller 根据 diff、改动位置、关键模块名、AST 变化做的自动归类
- `proposal_family`：最终用于统计和调度的 canonical family

默认规则：

- 若两者一致，则直接采用
- 若两者冲突，则优先采用 `controller_inferred_family`
- 若 controller 也无法稳定归类，则标记为 `uncertain`

这样可以避免 worker 为了迎合 prompt，把跨 family 的改动误报成“champion 邻域微调”。

## 4. 目标数据结构改造

### 4.1 `results.tsv` 扩展字段

建议把 [results.tsv](/Volumes/KINGSTON/work/research/AutoNeoAg/results.tsv) 的 schema 扩展为：

- 现有字段全部保留
- 新增以下字段：
  - `worker_declared_family`
  - `worker_declared_subfamily`
  - `controller_inferred_family`
  - `controller_inferred_subfamily`
  - `proposal_family`
  - `proposal_subfamily`
  - `family_consensus`
  - `parent_round_id`
  - `parent_commit`
  - `search_mode`
  - `delta_vs_best`
  - `delta_vs_parent`
  - `novelty_level`
  - `decision_reason`
  - `failure_mode`
  - `confirm_checked`
  - `confirm_round_score`
  - `confirm_survival`

推荐顺序：

1. `task_id`
2. `strategy`
3. `run_id`
4. `round_id`
5. `commit`
6. `worker_declared_family`
7. `worker_declared_subfamily`
8. `controller_inferred_family`
9. `controller_inferred_subfamily`
10. `proposal_family`
11. `proposal_subfamily`
12. `family_consensus`
13. `parent_round_id`
14. `parent_commit`
15. `search_mode`
16. `dev_score`
17. `confirm_score`
18. `blind_score`
19. `confirm_checked`
20. `confirm_round_score`
21. `confirm_survival`
22. `delta_vs_best`
23. `delta_vs_parent`
24. `status`
25. `decision_reason`
26. `failure_type`
27. `failure_mode`
28. `training_seconds`
29. `lines_changed`
30. `novelty_level`
31. `description`

### 4.2 字段语义

#### `worker_declared_family` / `controller_inferred_family` / `proposal_family`

三者的语义不同：

- `worker_declared_family`：worker 自报
- `controller_inferred_family`：controller 推断
- `proposal_family`：最终 canonical family，用于统计和调度

建议初始集合：

- `preference_context`
- `preference_contrast`
- `scalar_contrast`
- `interaction_balance`
- `gating`
- `ranking_objective`
- `auxiliary_head`
- `fusion_path`
- `other`
- `uncertain`

#### `worker_declared_subfamily` / `controller_inferred_subfamily` / `proposal_subfamily`

家族内更细的结构标签，例如：

- `shared_competition_embedding`
- `joint_preference_head`
- `scalar_context_encoder`
- `preference_delta_refiner`
- `contrast_hla_gate`

#### `family_consensus`

建议值：

- `agreed`
- `controller_override`
- `worker_only`
- `controller_only`
- `uncertain`

#### `parent_round_id`

表示这轮 proposal 是沿哪个已知候选继续优化。  
如果本轮直接沿 champion 优化，则填 champion 的 round id。

#### `search_mode`

值域固定为：

- `exploit`
- `explore`
- `recovery`

#### `delta_vs_best`

本轮 `dev_score - 当前 best_score(提案前)`

#### `delta_vs_parent`

本轮 `dev_score - parent_round_score`

#### `novelty_level`

建议使用离散级别：

- `low`
- `medium`
- `high`

规则：

- `low`：在 champion family 中做一阶邻域微调
- `medium`：切到相邻 family 或引入一个新子模块
- `high`：新 family、目标函数切换、较大结构迁移

#### `decision_reason`

替代当前“只有 keep/discard”的黑箱决策。建议值：

- `new_best`
- `near_tie_but_worse`
- `clear_regression`
- `train_failed`
- `worker_failed`
- `no_op`
- `family_frozen`

#### `failure_mode`

更细颗粒度的失败分类，区别于 `failure_type`：

- `over_conditioned_head`
- `family_repeat_regression`
- `unstable_training`
- `redundant_context_injection`
- `oversized_edit`
- `unknown`

#### `confirm_checked` / `confirm_round_score` / `confirm_survival`

这三列用于低频 confirm 回灌，而不是每轮都跑 confirm：

- `confirm_checked`：本轮是否触发 confirm
- `confirm_round_score`：若触发 confirm，本轮 confirm 分数
- `confirm_survival`：本轮 proposal 是否通过 confirm gate

## 5. proposal family 体系

### 5.1 初始 family taxonomy

建议 controller 与 worker 共用一份 family 枚举，但 controller 对 canonical family 有最终解释权：

#### `preference_context`

特征：

- 构造 shared competition embedding
- 在 mut / wt preference 输入前追加上下文
- 对 `preference_delta_hidden` 的来源做轻量上下文增强

#### `preference_contrast`

特征：

- 新增或改造 joint preference contrast head
- 直接处理 mut / wt preference state 的 delta / product / mean
- 围绕 `preference_contrast_head` 做轻量改造

#### `scalar_contrast`

特征：

- 修改 `scalar_contrast_head`
- 改 comparison/context scalar block 的条件化方式
- 在 scalar path 中引入 competition context

#### `interaction_balance`

特征：

- 显式建模 `mut_hla` / `wt_hla` / `mut_hla - wt_hla`
- 新增 interaction summary / balance head

#### `gating`

特征：

- gate mut / wt state
- gate preference delta
- gate scalar or context branch

#### `ranking_objective`

特征：

- 切换或混入 `pairwise` / `group` / `hybrid_pairwise`
- 对 loss/objective 做主导性调整

#### `auxiliary_head`

特征：

- 新增独立 auxiliary logit 或独立支路
- 不属于 champion 主干的一次性小 head

#### `fusion_path`

特征：

- 修改主 contrast / final conditioning 的融合路径
- 做轻量 residual / normalization / affine modulation

### 5.2 当前 `neoantigen` 轨迹的 family 映射建议

根据 [results.tsv](/Volumes/KINGSTON/work/research/AutoNeoAg/results.tsv) 当前可初步映射：

- `round 17` -> `preference_context`
- `round 19` -> `preference_contrast`
- `round 20` -> `gating`
- `round 21` -> `preference_contrast`
- `round 22` -> `scalar_contrast`
- `round 24` -> `preference_contrast`
- `round 29` -> `interaction_balance`
- `round 39` -> `fusion_path`

这个映射不要求一开始 100% 完美，但必须可重复执行。  
优先做“稳定可分析”，再追求“家族定义精细”。

### 5.3 controller family 推断建议

controller 的 family 推断不需要一开始就做复杂语义理解，但至少应结合：

- 改动集中在哪些类、函数、模块名
- diff 中是否出现 `gate`、`contrast`、`preference`、`pairwise`、`rank` 等关键字
- 是新增模块、替换输入，还是切换 objective
- AST 层面是否新增 head / encoder / block

第一版可接受“规则表 + 关键字 + 改动位置”的推断逻辑。  
后续如果需要，再升级成更稳的 AST / diff classifier。

## 6. Frontier Summarizer 设计

### 6.1 输出工件

每轮 proposal 前，controller 自动生成：

- `artifacts/logs/<task>/<strategy>/run_<id>/frontier_state.json`
- `artifacts/logs/<task>/<strategy>/run_<id>/frontier_hint.md`
- `artifacts/logs/<task>/<strategy>/run_<id>/family_stats.tsv`

### 6.2 `frontier_state.json` 最小结构

建议结构：

```json
{
  "task_id": "neoantigen",
  "strategy": "constrained",
  "run_id": 1,
  "current_round": 43,
  "champion": {
    "round_id": 19,
    "commit": "45fa684",
    "dev_score": 0.759939,
    "proposal_family": "preference_contrast",
    "proposal_subfamily": "joint_preference_contrast_head"
  },
  "search_mode": "exploit",
  "confirm_feedback": {
    "enabled": true,
    "policy": "every_3_keeps_or_new_best",
    "last_checked_round": 43
  },
  "family_stats": [
    {
      "proposal_family": "preference_contrast",
      "attempts": 8,
      "keeps": 1,
      "best_gain": 0.014899,
      "recent_trend": "negative",
      "confirm_checks": 1,
      "confirm_promotions": 1,
      "confirm_survival_rate": 1.0,
      "frozen_until_round": null
    }
  ],
  "prioritize": [
    "preference_context",
    "preference_contrast",
    "fusion_path"
  ],
  "avoid": [
    "gating",
    "ranking_objective",
    "auxiliary_head"
  ],
  "recent_fail_patterns": [
    "direct context expansion into existing preference_contrast_head",
    "repeat scalar branch conditioning without new structural change"
  ]
}
```

### 6.3 `family_stats.tsv` 字段

建议包含：

- `proposal_family`
- `attempts`
- `keeps`
- `mean_delta_vs_best`
- `best_gain`
- `last_gain_round`
- `recent_5_mean`
- `consecutive_regressions`
- `confirm_checks`
- `confirm_promotions`
- `confirm_survival_rate`
- `frozen_until_round`

### 6.4 `frontier_hint.md` 内容模板

内容不宜过长，建议固定结构：

1. Current champion
2. Successful patterns
3. Repeated failure patterns
4. Priority for this round
5. Avoid for this round
6. Parent round and expected search mode

示例：

```md
# Frontier Hint

Current champion: round 19, family preference_contrast, dev_score 0.759939.

Successful patterns:
- HLA-conditioned preference competition
- WT-vs-Mut contrast refinement
- Shared scalar competition context before preference delta

Recent failure patterns:
- Directly expanding existing preference_contrast input with more raw context
- Repeated scalar branch conditioning without new structural role
- Gating variants that destabilize the preference path

Priority for this round:
- Refine champion branch with one small residual or normalization change
- Improve coupling between champion branch and main contrast path
- Keep the edit small and local

Avoid:
- objective switch
- new large auxiliary head
- gating-heavy edits
- revisiting a failed family for the third time
```

## 7. controller 具体改造

### 7.1 新增模块建议

建议新增：

- `src/autoneoag/runtime/frontier.py`

负责：

- 从 `results.tsv` 读取历史
- 归类 family
- 统计 family 表现
- 决定 `search_mode`
- 生成 `frontier_state.json`
- 生成 `frontier_hint.md`
- 生成 `family_stats.tsv`

### 7.2 controller 新流程

当前流程：

1. 读取 `summary_lines`
2. 调用 worker
3. 训练
4. 记录结果

改造后流程：

1. 读取当前 run 的历史结果
2. 用 `frontier.py` 生成 frontier state
3. 决定本轮 `search_mode`
4. 将 `frontier_hint` 与 champion 元信息传给 worker
5. worker 返回带 family 的 proposal
6. controller 基于 diff 推断 `controller_inferred_family`
7. 合并 worker / controller family，生成 canonical `proposal_family`
8. 训练
9. 计算 `delta_vs_best` / `delta_vs_parent`
10. 若命中低频 confirm 策略，则额外跑 confirm
11. 写入扩展后的 `results.tsv`
12. 更新 family 状态和冻结状态

### 7.3 family 冻结规则

建议默认启用，但明确把它们视作 **v1 启发式**，不是跨任务的最终规律：

- 同一 family 连续 `3` 轮 `delta_vs_best <= -0.03`  
  -> 冻结 `5` 轮

- 连续 `8-10` 轮没有新 best  
  -> 当前主 family 降权，允许切换到次优 family

- 若某 family 最近 `5` 次尝试中：
  - `keeps = 0`
  - `mean_delta_vs_best < -0.05`
  -> 标记为 `avoid`

实现要求：

- 这些阈值放进配置，不要硬编码在逻辑里
- `frontier.py` 保留后续升级到 family score / bandit / UCB 风格调度的接口
- 当前目标是先稳定减少重复无效搜索，而不是一次性做最优策略学习

### 7.4 搜索模式切换规则

建议：

- 默认：`exploit`
- 若 champion family 已冻结：切到 `explore`
- 若连续 `10` 轮无新 best：切到 `recovery`
- 若最近两轮都是 `train_failed` 或 `worker_failed`：切到 `exploit`，并缩小 edit 半径

### 7.5 低频 confirm 回灌

为了避免 family 评价只围绕 `dev_score`，建议加入低频 confirm 回灌：

- 默认不在每轮触发 confirm
- 对每个 `new_best` 触发一次 confirm
- 或者每累计 `3` 个 `keep`，触发一次 confirm

记录方式：

- 把 confirm 结果写回 `results.tsv`
- 把 family 的 `confirm_survival_rate` 写入 `family_stats.tsv`
- 在 `frontier_hint.md` 里提示哪些 family 虽然 dev 上涨，但 confirm 存活率偏低

这样 family 的“优先级”会同时参考：

- dev 改进
- recent trend
- confirm survival

## 8. worker 与 schema 改造

### 8.1 `codex_worker_output.schema.json` 扩展

建议新增字段：

- `worker_declared_family`
- `worker_declared_subfamily`
- `proposal_family`
- `proposal_subfamily`
- `parent_round_id`
- `search_mode`
- `novelty_level`

推荐 schema：

- 保留原有：
  - `hypothesis`
  - `expected_change`
  - `risk`
  - `edit_scope`
  - `summary`
- 新增：
  - `worker_declared_family`
  - `worker_declared_subfamily`
  - `proposal_family`
  - `proposal_subfamily`
  - `parent_round_id`
  - `search_mode`
  - `novelty_level`

### 8.2 worker prompt 改造原则

[src/autoneoag/runtime/codex_worker.py](/Volumes/KINGSTON/work/research/AutoNeoAg/src/autoneoag/runtime/codex_worker.py) 当前 prompt 只给最近摘要。  
改造后应改成：

- 给出 `Current champion`
- 给出 `Successful patterns`
- 给出 `Recent failure patterns`
- 给出 `Priority`
- 给出 `Avoid`
- 给出 `search_mode`
- 给出 `parent_round_id`

并要求 worker：

- 显式标注 `worker_declared_family`
- 如果是 `exploit`，只能做 champion family 的一阶邻域改动
- 如果是 `explore`，必须说明为什么切到相邻 family
- 如果是 `recovery`，必须说明为什么当前 champion family 不值得继续重复
- 不要为了迎合 hint，把跨 family 改动伪装成 champion family 的小修补

### 8.3 prompt 约束模板

建议在 prompt 中固定加入：

- Only make one small structural change.
- Prefer refining the current winning family over introducing a new branch.
- Avoid repeated failure patterns listed below.
- If you choose a new family, justify why the current frontier warrants exploration.

## 9. 分阶段实施计划

### Phase 1：结果 schema 升级

目标：

- 扩展 `results.tsv` header
- `append_result` 支持新增字段
- 允许旧结果向后兼容

交付物：

- `src/autoneoag/runtime/results.py`
- 迁移脚本或兼容逻辑

验收标准：

- 新 run 能写入扩展字段
- 旧 `results.tsv` 仍可读取

### Phase 2：family 归类与 frontier 输出

目标：

- 新增 `frontier.py`
- 每轮落地 `frontier_state.json`
- 每轮落地 `frontier_hint.md`
- 每轮落地 `family_stats.tsv`

验收标准：

- 在已有 `neoantigen` 历史上可生成稳定 family 统计
- 当前 champion 和 avoid family 能被正确识别
- worker 自报与 controller 推断的冲突能被明确记录

### Phase 3：worker schema 与 prompt 改造

目标：

- worker 输出带 family / search mode / parent round
- prompt 消费 `frontier_hint`

验收标准：

- 连续 5 轮 worker 都能返回合法 family 标签
- family 标签与人工判断基本一致
- 出现冲突时，controller 能覆盖 worker 自报并留下 `family_consensus`

### Phase 4：controller exploit / explore / avoid 调度

目标：

- controller 自动决定本轮 `search_mode`
- family 冻结机制生效
- repeated failure pattern 生效

验收标准：

- 相同失败 family 不会连续无限重试
- 当连续多轮无新 best 时，会自动切换主方向
- 低频 confirm 能对 family 优先级产生可追踪影响

### Phase 5：回放验证

目标：

- 用现有 `neoantigen` 轨迹回放
- 验证新机制会不会比 blind search 更快收敛

建议做法：

- 用 `round 1-20` 的历史构造离线 frontier
- 检查新规则是否会：
  - 更早集中到 `round 17/19` 这类方向
  - 更早冻结 `gating`
  - 减少无效 `scalar_contrast` 重复尝试
  - 在 `hla_immunogenicity` 上更早识别“长期平台但接近 best”的阶段

验收标准：

- 离线分析显示无效 family 重复率下降
- champion family 被更多 exploit 轮次覆盖

## 10. 当前任务上的直接策略建议

基于 2026-04-01 的当前 `neoantigen` 全量轨迹，推荐默认策略：

### 10.1 优先家族

- `preference_context`
- `preference_contrast`
- `fusion_path`

### 10.2 降权家族

- `scalar_contrast`
- `interaction_balance`

### 10.3 临时避免家族

- `gating`
- `ranking_objective`
- `auxiliary_head`

### 10.4 当前建议的 exploit 方向

- champion branch 的轻量 residual/refinement
- champion branch 与 main contrast 的更稳融合
- champion branch 的 normalization / affine conditioning

### 10.5 当前建议避免的模式

- 把更多 raw context 直接硬塞进已有 `preference_contrast_head`
- 新增独立 auxiliary head
- 纯 `gating` 变体
- 重新切回 ranking objective

## 11. 风险与控制

### 风险 1：family 分类过于主观

控制：

- 第一版允许粗分类
- 分类规则写死在 `frontier.py`
- 允许 `other`

### 风险 2：历史偏见导致过早收缩

控制：

- 保留 `20%` explore 配额
- 连续若干轮无新 best 时自动切到 `recovery`

### 风险 3：prompt 过长反而稀释重点

控制：

- `frontier_hint.md` 限制在 15-25 行
- 只给最近最重要的 family 结论

### 风险 4：schema 变更打断当前实验兼容性

控制：

- `results.py` 提供向后兼容读取
- 新字段缺失时自动补默认值

## 12. 最小可执行版本

如果只做一个最小版本，建议只实现以下六件事：

1. `results.tsv` 新增 `proposal_family`、`search_mode`、`parent_round_id`
2. 同时新增 `worker_declared_family` 与 `controller_inferred_family`
3. 新增 `frontier_hint.md`
4. worker prompt 使用 `frontier_hint.md`
5. controller 加入 family 冻结规则
6. 对 `new_best` 做低频 confirm 回灌

这样就已经能从 blind search 升级到基础版 direction-aware search。

## 13. 结论

这套机制的意义不是“让 agent 记住所有历史”，而是让它具备三种能力：

- 记住什么方向有效
- 记住什么方向不值得重复
- 记住当前最值得优化的是哪条 frontier

但它不应该把“研究记忆”做成“研究偏见”。  
因此第一版实现必须同时满足：

- family 标签有双轨来源，而不是只信 worker 自报
- 冻结和调度规则是可配置的启发式，而不是不可质疑的硬规则
- family 优先级不仅看 dev，还要留下低频 confirm 存活证据

如果当前 AutoNeoAg 要往 framework 论文推进，这套机制值得优先实现，因为它直接对应论文里的核心主张：

**agent 不是自由试错，而是在可审计轨迹上形成研究记忆，并据此调整后续搜索方向。**
