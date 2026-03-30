# plan

这份文档整合了 `plan_pre.md` 中仍然成立的背景、动机与执行设想，并以当前这个更严格的版本作为主计划文档。

## 0. 为什么这个方向值得做

这个方向命中了一个真实且尚未解决的临床瓶颈：**新抗原免疫原性预测的正预测率太低**。TESLA 联盟显示，即便是排名最靠前的候选肽，真正被实验确认为免疫原性的比例也很低；从更广义的流程看，绝大多数计算预测的新抗原最终并不能稳定进入“可用于疫苗或 T 细胞验证”的候选集合。换句话说，问题不在于“能不能提出很多候选”，而在于“能不能把真正值得验证的候选排到 top-k”。这正是一个很适合方法学优化的子问题。

同时，这个方向也和 `autoresearch` 的核心范式高度契合。免疫原性预测并不是一个必须依赖超大模型和超长训练周期的问题；相反，它更像一个**小数据、高偏倚、高约束、但搜索空间很丰富**的任务。只要把数据清洗、切分和评估体系冻结好，agent 就可以围绕一个明确目标持续迭代。

## 0.5 为什么 `autoresearch` 适合这个问题

我保留 `plan_pre.md` 的基本判断，但把它放到更严格的实验设计下理解：

- **有自动化评估目标**：AUPRC、PPV@k、NDCG、TESLA 风格指标都可以程序化计算。
- **中小模型即可闭环**：树模型、小型 MLP/CNN/BiLSTM、轻量 Transformer 都有现实意义，不需要先上大规模基础模型训练。
- **搜索空间真实存在**：输入编码、特征组合、fusion 方式、损失函数、训练制度、多任务和预训练策略都可以系统搜索。
- **领域知识可以被显式注入**：`program.md` 不是附属文档，而是“研究组织代码”，是你把免疫学先验、风控规则和优先方向传达给 agent 的主要接口。

从 `plan_pre.md` 延续下来的一个重要观点我保留不变：这个项目最有价值的，不只是“能不能做出一个更强模型”，还包括**agent 在严格约束下到底会保留什么设计、会否定什么常识、会暴露哪些 shortcut**。

我会把原方案优化成一个更“硬”、更容易投稿、也更适合 `autoresearch` 的版本：

**不是做一个泛泛的“新抗原分类器”，而是做一个“在严格防泄漏设置下，自动进化人类 HLA-I 新生抗原 top-k 排序模型”的方法学研究。** 这么改的原因很直接：neoantigen 计算流程本来就包含 HLA typing、表达量、pMHC 呈递、TCR 识别等多步，最适合 `autoresearch` 的是其中一个可自动评估、单卡可闭环优化的子问题；而且 `autoresearch` 本身就是“单 GPU、固定时间预算、agent 只改 `train.py`、人类通过 `program.md` 约束研究方向”的模式。([OUP Academic](https://academic.oup.com/bib/article/26/4/bbaf302/8182748))

先说最关键的优化结论：**不要直接拿 TumorAgDB2.0 / NeoTImmuML 那套 10,312 样本去开跑。** 这套数据构建里已经把 TESLA 的 608 条序列纳入基础库，又额外纳入了 2024–2025 文献；NeoTImmuML 训练集还混入了 simulated positives、simulated negatives 和 mouse-validated neoantigens，并且作者自己明确承认其切分基于 sequence uniqueness 而不是 sequence similarity，存在 sequence-related bias 风险。更要命的是，这些 simulated labels 本身部分由 IC50 阈值规则生成；如果你再把 binding-related 特征喂给模型，模型很可能学到的是“标签生成规则”，不是免疫原性规律。([Frontiers](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2025.1681396/full))

所以我建议你按下面这套方案推进。

## 1. 研究问题先收窄

第一版只做 **human, HLA-I, short peptide, post-presentation ranking**。也就是说，不碰变异检测、不碰 HLA 分型、不碰全流程管线，只优化“给定候选 mutant peptide–HLA 对，如何把真正能诱导 T 细胞反应的候选尽量排到 top-k”。这和 TESLA、ImmuneApp-Neo、NeoRanking 一类工作的评价方式更一致，也更符合临床里“最后只会验证很少几个 top-ranked 候选”的现实。([科学直接](https://www.sciencedirect.com/science/article/pii/S0092867420311569))

更具体一点，我会把 v1 scope 冻结成：

- 仅做人类 MHC-I。
- 肽长先做 8–11 aa；12–13 aa 作为扩展，不进首版主分析。
- 目标定义为 **ranking**，不是单纯 binary classification。
- `autoresearch` 只优化这个 ranking model，不优化上游 calling/haplotype/presentation pipeline。([OUP Academic](https://academic.oup.com/bib/article/26/4/bbaf302/8182748))

## 2. 数据不要“一锅炖”，而要分层

我会把训练数据分成三层，并把这个分层写死在 `prepare.py` 里，agent 不能改。

**Tier A：strict human functional labels**

只保留“人类、MHC-I、明确功能实验阳性/阴性”的样本。阳性必须来自 ELISPOT、FACS、细胞因子释放等功能证据；阴性也尽量要求有明确的非免疫原性实验记录。**mouse、simulated、binding-only、presentation-only 全部排除**。这样做是因为 NeoTImmuML 的大数据版虽然有 10,312 个样本，但其中混入了 simulated 数据，而 TumorAgDB2.0 自身报告的 validated neoantigens 数量远低于这个量级，说明真正严格标签的数据会小很多，所以这本质上是个“小数据、高偏倚风险”的问题。([Frontiers](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2025.1681396/full))

**Tier B：curated-but-weaker labels**

可以纳入数据库里来源较好、但没有完备 patient-level functional context 的人类样本，作为辅助数据；训练时给较低 sample weight，不参与最终模型选择的核心指标。

**Tier C：weak/simulated labels**

包括 simulated positives、simulated negatives、dbSNP 派生 decoys、mouse 数据等。这一层**不进入主结论**，只能做两件事：

第一，做 encoder 预训练；

第二，做 auxiliary task。

而且如果这层标签本身由 binding 阈值构造，就**不要同时喂 raw binding 特征进主任务头**，否则会出现 shortcut learning。近年的免疫机器学习文献已经反复提醒，负样本构造方式会严重影响模型泛化。([Frontiers](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2025.1681396/full))

我会把数据表做成至少这些列：

```
peptide_mut, peptide_wt, hla, gene, aa_change, study_id, patient_id,
assay_type, label, label_tier, source_year, source_name,
is_tesla, is_2024plus, is_simulated, is_mouse
```

## 3. 切分方案是这个课题成败的核心

原 proposal 最大的问题之一，是默认可以“每轮都看 TESLA”。这不行。TESLA 608 条里只有 37 条阳性，本来就很小；一旦让 agent 连续几百轮盯着它调，TESLA 就不再是 test set，而是被你偷偷用成了巨量 validation set。([科学直接](https://www.sciencedirect.com/science/article/pii/S0092867420311569))

我会用 **三层评估**：

**内层开发集**

用于 `autoresearch` 每轮返回 `val_score`。这是 agent 唯一能看到的反馈。

**确认集**

仍来自训练语料范围内，但切分更严格，用于人类每天/每两天确认“leaderboard 是否真实”。

**外层 lockbox**

真正盲测，agent 永远看不到。至少两个：

1. **TESLA lockbox**：如果要用 TESLA 做最终 benchmark，就必须先把 TumorAgDB2.0 里所有 TESLA 来源样本彻底剔除。
2. **时间外推 lockbox**：如果要用 2024–2025 文献做独立测试，就必须先从训练集中剔除这批年份和来源，不能再直接沿用 NeoTImmuML 那种把 2024–2025 数据纳进数据库再“独立测试”的思路。([Frontiers](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2025.1681396/full))

内层切分我建议这样做：

1. exact dedup：相同 peptide–HLA–label 全删重。
2. mutation-event grouping：同一突变事件的 WT/Mut 对绝不跨 fold。
3. similarity clustering：同 HLA 下，任意共享长 k-mer 或高序列相似的肽聚成一组。
4. study-aware grouping：同一 study / patient 尽量不跨 fold。
5. StratifiedGroupKFold：按 label、peptide length、HLA supertype 分层。

此外再加两个 challenge split：

- **leave-study-out**
- **leave-HLA-supertype-out**

这样最后 reviewer 才很难说你只是记住了 assay/source/length bias。这个担心不是空穴来风：NeoTImmuML 自己就明确写了 similarity-based split 是未来工作，而它当前最重要的 SHAP 特征里，peptide length 本身就是最强特征之一。([Frontiers](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2025.1681396/full))

## 4. 指标体系不要只看 AUC

这个任务本质是“top-k 候选排序”。所以我建议：

**内层 `val_score` 用一个固定标量，但报告全部子指标。**

我会用这个版本：

```python
val_score = (
    0.45 * AUPRC +
    0.35 * PPV20 +
    0.10 * PPV10 +
    0.10 * NDCG20
)
```

原因是：

- AUPRC 比 AUROC 更适合低阳性率任务；
- PPV20 / PPV10 更贴近临床只验证 top 候选的场景；
- NDCG20 能看排序前列的整体质量。([Nature](https://www.nature.com/articles/s41467-024-53296-0))

**TESLA 外层评估则直接用官方风格指标**：

- AUPRC
- FR
- TTIF

TTIF 的官方实现就在 ParkerICI 的 TESLA repo 里。你内部可以用 PPV20 作为更通用的近似指标，但最终论文里 TESLA 部分最好直接跑官方函数。([GitHub](https://github.com/ParkerICI/tesla/blob/master/performance-metric-functions.R))

另外一定要加三类分层报告：

- per-length
- per-HLA supertype
- per-study / per-assay

否则很容易出现“整体涨了，其实只是在某个长度桶里投机”的假进步。NeoTImmuML 对 length、hydrophobicity、non-polar composition 的 SHAP 结果，和 IMPROVE 对 hydrophobic/aromatic residue 的结果，都说明这些特征既可能是真信号，也可能成为 shortcut，所以必须分层检查。([Frontiers](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2025.1681396/full))

## 5. baseline 一定要先做强，再让 agent 上场

这里我会很保守。第一周末之前，不让 agent 搜索，只做强 baseline leaderboard。

最低配置我建议这 8 个：

1. NetMHCpan / presentation-only score
2. Logistic regression on handcrafted features
3. Random Forest
4. XGBoost
5. LightGBM
6. MLP on fixed scalar features
7. sequence + HLA 小型 CNN / BiLSTM
8. scalar + sequence fusion MLP/CNN

这么排是因为当前公开工作已经说明：

ImmuneMirror 用 balanced random forest 能做到测试 AUC 0.87；NeoTImmuML 的前三强也是 LightGBM、XGBoost、Random Forest，再做 weighted ensemble；而最近更强的路线才开始转向 transfer learning、多任务和 multimodal structure。也就是说，你不先把树模型和小网络做扎实，后面 agent 找到的“提升”很可能只是战胜了一个太弱的起点。([OUP Academic](https://academic.oup.com/bib/article/25/2/bbae024/7606138))

这一步还有一个工程上的关键点：由于 `autoresearch` 明确限制 agent 不能新增依赖，你要在项目初始环境里就把 `scikit-learn`、`xgboost`、`lightgbm`、PyTorch 这些依赖配好，否则 agent 后面根本摸不到这一类强 baseline。([GitHub](https://github.com/karpathy/autoresearch/blob/master/program.md?plain=1))

## 6. `autoresearch` 框架怎么改才真正可用

我建议保持 Karpathy 的核心精神不变：`prepare.py` 固定，`train.py` 是 agent 唯一可改文件，人类只改 `program.md`。但在 repo 里多放两个**固定文件**：

- `confirm.py`：人类用来复核 top commits
- `blind_eval.py`：只在 lockbox 上跑，不给 agent 看标签

这样既保留单文件搜索的可审阅性，又把“快反馈”和“真验证”分开。([GitHub](https://github.com/karpathy/autoresearch/blob/master/program.md?plain=1))

文件职责我会这样定：

**`prepare.py`**

做人类冻结的所有事：

数据清洗、分层标签、split manifest、特征缓存、embedding 缓存、指标函数、`val_score` 计算。因为 agent 后面不能改它，所以这里必须一次性把未来可能有用的输入都准备好，包括：

- mutant peptide token
- WT peptide token
- HLA pseudosequence
- length / hydrophobicity / aromaticity / non-polar ratio
- binding / EL rank / stability
- expression / agretopicity / foreignness
- frozen pLM embedding
- metadata（study、patient、tier、assay）

这些方向都和现有工作一致：TESLA/NeoRanking 强调 presentation + recognition 特征，ImmuneApp-Neo 证明 presentation 预训练有帮助，NeoaPred / ImmunoStruct 说明结构与 foreignness 也是可扩展方向。([科学直接](https://www.sciencedirect.com/science/article/pii/S0092867420311569))

**`train.py`**

只负责 agent 搜索。第一版我会把搜索空间限制成 5 类：

- 特征门控：用哪些输入、怎么 fusion
- 模型家族：MLP / CNN / BiLSTM / 小 Transformer / cross-attention
- 损失函数：BCE / focal / pairwise ranking / listwise ranking
- 训练制度：strict-only、weak-pretrain→strict-finetune、多任务
- 简单集成与校准：temperature scaling / small ensemble

这里把 `plan_pre.md` 里比较宽的“可探索空间”收紧为更可控的版本，但保留其核心内容。可供 agent 调用或组合的输入候选，原则上包括：

- 肽段轻量编码：one-hot、BLOSUM62、可学习 token embedding
- HLA 表征：HLA pseudosequence、supertype、可学习 allele embedding
- 固定标量特征：binding / EL rank / stability / length / hydrophobicity / aromaticity / non-polar ratio / agretopicity / foreignness / expression
- 冻结表示：frozen protein LM embedding 或其他提前缓存好的表征

这些输入是否真正有用，不由先验决定，而由 strict split 下的验证结果决定。

多任务和 presentation→immunogenicity transfer 是值得重点试的，因为最近的模型路线已经在往这边走：ImmuneApp-Neo 通过 presentation 预训练再迁移到 immunogenicity，PPV 明显提升；UnifyImmun 则把 pHLA 与 pTCR 结合成联合学习框架。([Nature](https://www.nature.com/articles/s41467-024-53296-0))

**`program.md`**

我建议你一开始就写得比原 repo 更具体。可以直接这样起：

```markdown
# AutoNeoAg Program

Goal:
Maximize grouped-validation ranking quality on STRICT human HLA-I data.
Primary objective is PPV20/AUPRC, not raw AUROC.

Hard constraints:
- Never touch external lockboxes (TESLA / time-split holdout)
- Never optimize on weak-label-only gains
- Any gain driven only by peptide length is suspicious
- Simpler models win ties
- No feature may be introduced that leaks label construction rules

Priority directions:
1. strict-only strong baselines
2. weak-pretrain -> strict-finetune
3. pairwise/listwise ranking losses
4. WT-vs-Mut delta modeling
5. multitask heads for presentation/stability/immunogenicity

Required reporting:
val_score, AUPRC, PPV10, PPV20, NDCG20,
per-length breakdown, peak_vram_mb, params
```

## 7. 6 周的可落地执行计划

### 第 1 周：数据审计与锁箱冻结

交付物：

- `data_card.md`
- `source_manifest.csv`
- `lockbox_manifest_tesla.csv`
- `lockbox_manifest_timesplit.csv`
- `strict_train_candidates.csv`

必须完成的检查：

- 所有 TESLA 来源样本从训练语料中删除
- 所有 2024–2025 holdout 来源样本从训练语料中删除
- human / mouse / simulated / curated / assay type 全部打标签
- exact duplication 与近重复规则确定并冻结

### 第 2 周：特征缓存与 baseline leaderboard

交付物：

- `features.parquet`
- `embeddings.npy`
- `splits_grouped_v1.json`
- baseline 报告一份

要完成的模型：

- presentation-only
- logistic / RF / XGBoost / LightGBM
- MLP
- small CNN or BiLSTM
- fusion baseline

这一周结束时，你应该已经知道：

到底是 hand-crafted features 占优，还是 sequence branch 已经有额外信号；如果 sequence model 连树模型都打不过，就别急着上复杂 agent 搜索。([OUP Academic](https://academic.oup.com/bib/article/25/2/bbae024/7606138))

### 第 3–4 周：启动 autoresearch

节奏我会设成这样：

- 第 1 晚：只在 baseline 附近做微调，确认框架稳定
- 第 2–3 晚：放开 loss / fusion / weighting
- 第 4–5 晚：试 weak-pretrain 和 multitask
- 第 6 晚后：压缩搜索空间，只围绕 top 2–3 条路线深化

每天早上审查这几个问题：

1. 提升是否跨 fold 一致？
2. 提升是否只出现在某个 peptide length？
3. 提升是否只出现在某个 study / HLA supertype？
4. strict set 上涨了，weak set 是否只是噪声？
5. 新模型是否只是更大、更慢，而不是更好？

### 第 5 周：确认与盲测

对 top 10 commits 做：

- 5 seeds × grouped 5-fold confirm
- leave-study-out confirm
- leave-HLA-supertype-out confirm
- blind TESLA eval
- blind time-split eval

统计分析：

- AUPRC：bootstrap CI
- PPV10 / PPV20 / TTIF：paired bootstrap
- per-patient / per-study blocked bootstrap
- 最终只汇报“盲测前锁定”的 champion

### 第 6 周：写作与科学分析

论文主线不要只写“模型涨了多少”，还要写：

- agent 最终保留下来的设计是什么
- 哪些“常识设计”被 agent 否定了
- strict-only、weak-pretrain、多任务、结构扩展各自的边际贡献
- 去掉 length feature、去掉 binding feature 后性能还剩多少
- 在不同 HLA supertype / peptide length 上是否稳健

这一部分正是 agentic bioinformatics 的方法学价值所在。近期综述已经把 biological AI agents 视为一个独立上升方向，所以“受约束 agent 如何在高偏倚、小样本生物任务里搜索到可泛化模型”本身就是论文点。([OUP Academic](https://academic.oup.com/bib/article/26/5/bbaf505/8266996))

## 8. 预期产出与发表点

这一版计划吸收 `plan_pre.md` 的一个核心优点：它不只是一个工程项目，也确实有论文故事。

我预期至少有两层产出：

1. **模型结果本身**
   在 strict grouped-CV、leave-study-out、leave-HLA-supertype-out、blind TESLA、blind time-split 上，给出和 presentation-only、树模型、轻量深度模型的系统对比。
2. **autoresearch 过程洞察**
   记录 agent 真正保留了什么设计，哪些“看起来合理”的特征或架构在盲测中失效，哪些提升其实只是来源偏倚、长度偏倚或标签构造偏倚。

如果 blind external 上有稳定正向结果，这会是一篇“受约束 agent 搜索能够改善 neoantigen top-k ranking”的正结果方法学论文；如果 blind external 不涨，但能系统揭示 shortcut 和 split 设计的重要性，也仍然是一篇有价值的负结果方法学论文。

## 9. 资源与执行边界

这里也整合 `plan_pre.md` 中仍然有效的资源估算，但改得更保守一些。

- 数据：公开数据库与文献整理是可行的，但真正可用于主结论的 strict functional labels 规模会明显小于“数据库总样本数”。
- 算力：完整大规模夜间搜索最好有可用 GPU；但数据审计、切分冻结、特征缓存、树模型 baseline 和轻量神经网络验证可以先在本地完成。
- 软件：PyTorch、scikit-learn、xgboost、lightgbm 这类依赖要在项目初始化时一次配齐，避免 agent 后期触碰不到强 baseline。
- 周期：6 周拿到一个严谨 pilot 是现实的；是否继续深挖，取决于第 2 周和第 4 周的门槛是否通过。

## 10. 我给你的 go / no-go 标准

我会设三个硬门槛：

**2 周门槛**

如果 strict grouped-CV 上，最强 baseline 仍然只是树模型，而 sequence/fusion 模型没有稳定超过它，就先别扩大搜索，优先把数据与 split 修干净。

**4 周门槛**

如果 agent 在 confirm set 上能赢，但一到 blind TESLA / time-split 就掉回 baseline，说明学到的是 source bias，不值得继续烧太多钱。

**继续投入的条件**

至少满足下面两条中的一条：

- blind external 上 PPV20 / TTIF 有稳定提升
- leave-study-out / leave-HLA-supertype-out 上仍保持优势

只要这两条都过，你这个课题就很值得深挖；过不了，也不算白做，因为你仍然能产出一篇很有价值的方法学负结果：**在 neoantigen 这种高偏倚任务里，agent 搜索最容易学到哪些 shortcut，怎样的 split 和标签设计才能避免假进步。** 这类问题正是这个领域近几年一直在强调的痛点。([Nature](https://www.nature.com/articles/s43018-023-00675-z))

我的一句话结论是：

**值得做，但必须把它从“直接拿数据库+TESLA 开跑”升级成“严格分层标签 + 相似性防泄漏切分 + 双层盲测 + 强 baseline 先行”的版本。**

按这套方案，6 周能拿到一个很像样的 pilot；如果 4 周时 blind external 也有正向信号，就可以认真往投稿推进。

下一步我建议直接进入最实操的一步：我可以把这套计划继续展开成一个**项目初始化清单**，包括 `prepare.py` 字段设计、split manifest 规则、`val_score` 实现和第一版 `program.md`。
