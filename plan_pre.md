# plan_pre

说明：这是前期草案，当前主计划与整合版请以 `plan.md` 为准。

Yangfei，经过详细调研，我给你提一个非常具体的课题方案。

---

## 课题：AutoNeoAg — 基于 Autoresearch 的肿瘤新抗原免疫原性预测模型自动进化

### 为什么选这个方向

这个方向命中了一个真实的、未解决的临床痛点。TESLA 联盟的数据显示，即便是排名最靠前的预测肽段，也仅有约 6% 被实验确认为免疫原性阳性。据估计，目前不到 3% 的计算预测新抗原能在肿瘤位点引发强有力的 T 细胞反应。以最常用的 NetMHCpan 为例，虽然 ROC-AUC 表现最好，但统计表明预测的抗原中不到 5% 能呈递到细胞表面，不到 1% 能被 MHC 分子结合并识别。

换句话说，**免疫原性预测的正预测率（PPV）极低，是个性化癌症疫苗从实验室到临床的核心瓶颈**——而这恰好是一个有明确数值指标、模型规模适中、非常适合 autoresearch 循环的问题。

### 为什么 autoresearch pattern 在这里极其合适

先看三个前提条件是否满足：

**明确的自动化评估指标** ✅ — 新抗原免疫原性预测有标准化 benchmark：AUC-ROC、AUC-PR、以及 TESLA 联盟定义的 TTIF（Top Twenty Immunogenic Fraction，top 20 预测中免疫原性肽的数量）。这些指标可以全自动计算，无需人工判断。

**单 GPU 可跑** ✅ — 这类模型参数量通常在百万级以下（相比 LLM 的数十亿参数），5 分钟一轮训练完全可行。甚至可以在 5 分钟内完成完整的 5-fold 交叉验证。

**丰富的搜索空间** ✅ — 编码策略（one-hot、BLOSUM、protein language model embeddings）、网络架构（CNN、BiLSTM、Transformer、GNN）、特征组合（binding affinity、stability、expression、foreignness、hydrophobicity）、损失函数（focal loss、contrastive loss）、数据增强策略……这些给 agent 提供了大量有意义的探索方向。

### 具体实施方案

### 第一阶段：数据准备（1-2 周）

**训练数据**：TumorAgDB2.0 包含 10,312 个样本，正负样本各 5,156 个，涵盖 IEDB、NCI、TESLA 等多个权威来源的新抗原免疫原性数据，整理截至 2025 年 5 月。这是目前最大的公开免疫原性标签数据集。

**测试数据**（不参与训练，用于评估 autoresearch 的每一轮）：

- TESLA 数据集：608 个新抗原序列，37 个被实验证实能结合患者匹配的 T 细胞
- Rosenberg 数据集：5 名受试者，共 246 个突变肽，其中 4%（11 个）为新抗原
- 另外还可构建一个 2024-2025 年文献独立测试集（n=1,086），与训练集进行了严格的序列去重

**特征提取**（写入 `prepare.py`，固定不让 agent 修改）：

- 肽段序列编码（one-hot + BLOSUM62 + ESM-2 embeddings）
- HLA pseudosequence 编码
- MHC binding affinity（NetMHCpan %rank）
- 肽段稳定性（NetMHCstabpan）
- 疏水性特征（GRAVY 指数、疏水残基比例）
- Foreignness score（与自身蛋白组的序列相似度）
- 突变基因表达量（TPM）

### 第二阶段：搭建 autoresearch 框架（1 周）

仿照 Karpathy 的三文件架构：

**`prepare.py`**（固定）：数据加载、特征矩阵构建、train/val split、评估函数。核心评估函数输出一个综合 metric：

```python
# 复合评估指标：兼顾区分能力和临床可用性
metric = 0.4 * AUC_PR + 0.3 * TTIF_normalized + 0.3 * precision_at_k
```

之所以用 AUC-PR 而非 AUC-ROC，是因为免疫原性数据极度不平衡（阳性率极低），AUC-PR 对此更敏感。TTIF 直接衡量 top 排名中真正免疫原性肽的富集程度，模拟临床上"只选 top 20 个候选肽做疫苗"的实际场景。

**`train.py`**（agent 自由修改）：初始版本包含一个基础模型——简单的 CNN + Fully Connected 架构，类似现有 NUCC 的设计。Agent 可以改的包括但不限于：

- 模型架构（换成 BiLSTM、attention mechanism、graph neural network）
- 输入编码方式的组合和权重
- 损失函数（cross-entropy → focal loss → contrastive loss）
- 优化器选择和学习率调度
- 数据增强（随机 mask 残基、混合负样本采样策略）
- 多任务学习设计（同时预测 binding + stability + immunogenicity）
- Ensemble 策略

**`program.md`**（人类迭代）：这是你注入领域专家知识的地方，也是整个项目最有价值的「研究代码」：

```markdown
# AutoNeoAg Research Program

## Domain Context
You are optimizing a neoantigen immunogenicity predictor. The goal is
to maximize the composite metric (AUC-PR + TTIF + Precision@K).

## Key Domain Knowledge
- Immunogenicity depends on: MHC binding, peptide stability, TCR
  recognition (foreignness), antigen processing, expression level
- Current SOTA achieves ~0.80-0.89 AUC; major gap is PPV
- Hydrophobicity (GRAVY, non-polar ratio) and peptide length are
  among top predictive features (SHAP analysis confirms)
- Contrastive learning has shown promise (ConBoTNet approach)
- Transfer learning from binding affinity → EL → immunogenicity
  works (ImmugenX modular approach)

## Promising Directions
- Multi-task learning: jointly predict binding, stability,
  immunogenicity (shared encoder, task-specific heads)
- Attention over residue-residue pair encoding (RPEMHC-style)
- Focal loss to handle extreme class imbalance
- Self-supervised pre-training on unlabeled peptide-MHC pairs

## Constraints
- Do NOT modify prepare.py
- Training + evaluation must complete within 5 minutes
- Report val_metric (the composite score) at the end
```

### 第三阶段：运行与分析（2-4 周）

启动 autoresearch 循环，预期节奏：

- 每轮 ~5 分钟，每小时 ~12 个实验
- 一晚 ~100 个实验
- 运行 3-5 个过夜周期，累计 300-500 个实验

每天早上审查 agent 的 commit log，观察它探索了哪些方向，更新 `program.md` 引导下一轮方向。

### 第四阶段：验证与发表（2-4 周）

将最终模型在所有 hold-out 测试集上评估，重点报告：

- 在 TESLA 数据集上的 TTIF 对比（直接与 TESLA 联盟 35 个团队的结果对比）
- 在 Rosenberg 数据集上的 precision@5/10/20
- ELISpot 体外验证数据与预测得分的相关性

### 为什么这个课题有发表价值

**新颖性**在于方法论层面的双重创新：不仅是一个新的免疫原性预测器，更是**第一个用 AI agent 自主进化出来的免疫原性预测模型**。你可以同时报告：

1. **模型本身的性能**——和 NetMHCpan、MHCflurry、DeepNeo、PRIME、neoIM、NeoTImmuML 等横向对比
2. **autoresearch 过程的洞察**——agent 发现了哪些人类研究者没想到的架构/特征组合？哪些"常识性"的设计选择被 agent 否定了？

近期综述指出，深度学习、迁移学习和多组学整合模型有望进一步提升免疫肽组学的准确性和可扩展性——你的工作正好回应了这个方向。

### 资源需求估算

- 硬件：一张消费级 GPU（RTX 3090/4090 即可，模型很小）
- 数据：全部公开可获取（IEDB、TumorAgDB2.0、TESLA）
- 软件：PyTorch + Claude Code 或 Codex
- 时间：从准备到第一版可投稿结果，约 6-10 周
- 费用：Claude Code API 费用约 $50-100/周

这个课题和你在免疫肽组学、HLA 分型、肿瘤分析自动化方面的积累非常契合——你对 HLA 数据和免疫肽谱的深度理解，正好可以写出高质量的 `program.md` 来引导 agent。你觉得怎么样？
