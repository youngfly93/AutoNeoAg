# Autoresearch for Bioinformatics: 从自动调参到可审计研究系统

## 摘要
生物信息学中的许多预测任务并不缺少模型，而是缺少能够在高偏倚、小样本、强泄漏风险条件下持续产生可信改进的研究系统。传统 AutoML 更擅长在固定特征和固定目标上搜索超参数，而大语言模型驱动的 agent 系统虽然具备提出新结构假设和自动改写代码的能力，却常常因为评估协议不稳定、盲测边界不清晰、实验过程不可审计而失去科学可信度。本文提出一种面向生物信息学的 constrained autoresearch 范式，其核心目标不是让 agent 直接“发现生物学真理”，而是把研究流程重构为一个由冻结证据层、受限搜索层和可追溯实验日志组成的可审计研究系统。在该框架中，数据清洗、去重、分组切分、confirm 集和 blind 集评估、keep/revert 规则以及结果记录由确定性代码固定管理，而 agent 仅被允许在受控搜索面内修改训练脚本，从而把自由探索限制在建模归纳偏置层。我们以 HLA-I 新抗原排序任务为案例，构建了一个本机可运行的原型系统，并将搜索空间从低层 pooling 与 loss 微调提升到特征块组织、WT-vs-Mut 对比头、HLA 条件偏好分支以及 pair/group ranking 目标等高层结构。原型实验显示，扩大 grouped-CV 开发集后，agent 搜索不再被假进步误导，而能够识别真正有效的 HLA 条件 WT-vs-Mut 对比建模改进。本文主张，`autoresearch` 对生物信息学的真正价值，不在于替代科学家做最终裁决，而在于把研究流程形式化为一个能够持续优化、严格审计并适合盲测验证的系统。

## Abstract
Many bioinformatics prediction tasks do not primarily lack models; they lack research systems that can generate trustworthy improvements under small-sample regimes, leakage risk, and biased labels. Traditional AutoML is effective at tuning hyperparameters within a fixed model family, while LLM-based agents are capable of proposing structural hypotheses and editing code, yet often fail to provide scientific reliability because the evaluation protocol is unstable, blind validation is not protected, and the search process is poorly auditable. We propose a constrained autoresearch paradigm for bioinformatics, in which the objective is not to let an agent directly derive biological conclusions, but to reformulate research as an auditable system composed of a frozen evidence layer, a restricted search layer, and a fully logged experiment loop. In this setting, data cleaning, deduplication, grouped splits, confirm and blind evaluation, keep/revert rules, and experiment logging are managed deterministically, while the agent is restricted to editing a controlled training surface. This design shifts search from unconstrained automation toward the discovery of modeling inductive biases. Using HLA-I neoantigen ranking as a case study, we build a local prototype and expand the search space from low-level pooling and loss tweaks to higher-level modules such as feature-block organization, WT-vs-Mut contrast heads, HLA-conditioned preference branches, and pair/group ranking objectives. Pilot experiments show that once the grouped-CV development set is strengthened, the agent is less likely to chase false gains and is instead able to identify genuinely useful HLA-aware WT-vs-Mut contrast structures. We argue that the major contribution of autoresearch to bioinformatics is not autonomous scientific judgment, but the transformation of research itself into an optimizable, auditable, and blind-test-ready system.

## 1. 引言
近年来，生物信息学中的许多关键任务逐渐呈现出相似的结构性难题。以新抗原排序、致病变异优先级排序、肽段-HLA 结合后免疫原性建模以及多组学标志物优先级排序为代表，这类问题往往同时具有四个特点：第一，标签昂贵且稀疏；第二，数据来源异质且容易出现 study bias；第三，训练集、开发集与外部验证集之间的泄漏风险高于普通机器学习任务；第四，模型改进常常来自对领域结构的重新编码，而不是简单的超参数搜索。传统 AutoML 在这类问题上通常只能回答“在既定建模框架下哪个超参数更优”，却很难持续探索“哪种归纳偏置才更符合生物学结构”。

与此同时，基于大语言模型的 agent 系统展示了另一种潜力。与传统 AutoML 不同，agent 不仅能够调参，还能够读代码、改代码、提出结构性假设、运行实验并记录结果。这种能力使其天然适合承担研究型编排角色。然而，如果直接把 agent 置于开放研究环境中，它也会快速暴露出一个核心问题：一旦数据切分、评估方式、盲测集边界和实验记录规则本身都可被 agent 影响，那么系统优化的就不再是科学目标，而是漏洞利用和代理目标偏差。换言之，agent 的能力越强，研究系统对“证据主权”的要求就越高。

本文提出，`autoresearch` 在生物信息学中的意义，不应被理解为“让 AI 自动做科研”，而应被理解为“把科研流程重构为一个可审计、可复现、可盲测的自动化研究系统”。在这一系统中，评估协议必须先被冻结，agent 才被允许进入搜索环节；数据清洗、分组切分、confirm 集和 blind 集的定义必须先稳定，模型搜索才有意义；实验日志、代码版本和 keep/revert 规则必须先显式化，研究结论才具备追溯性。基于这一立场，本文尝试回答三个问题。第一，什么样的设计哲学能够使 autoresearch 真正适用于高偏倚的生物信息学任务。第二，agent 在这种系统中最适合承担什么角色。第三，这种设计是否能够在真实案例中发现比低层微调更有价值的建模归纳偏置。

本文的主要贡献有三点。其一，我们提出了一个面向生物信息学的 constrained autoresearch 框架，把“证据层冻结”和“搜索层受限”作为系统第一原则。其二，我们将这一框架具体化为一个由 `prepare`、`train`、`confirm`、`blind_eval`、`controller` 和研究约束文档组成的原型系统，并把搜索面限制在训练脚本内部。其三，我们以 HLA-I 新抗原排序为案例，展示在 grouped-CV 开发协议加强后，agent 搜索开始从低收益微调转向高层 WT-vs-Mut 对比建模，并产生了可重复的性能提升。

## 2. 从自动调参到研究系统
如果把现有自动化方法粗略划分，可以得到两种典型范式。第一种是 AutoML 范式，其基本单元是“给定模型族和参数空间，在验证集上寻找最优配置”。第二种是开放 agent 范式，其基本单元是“给定模糊目标和工具集合，让 agent 自主探索、修改、运行并汇报结果”。前者的问题在于搜索空间过窄，很难触及真正重要的生物学归纳偏置；后者的问题在于约束过弱，一旦评估协议本身被 agent 接触，就会出现目标漂移和不可审计性。

本文主张应在两者之间引入第三种范式，即研究系统范式。在这一范式中，系统的第一目标不是立即得到更好的模型，而是先建立一个谁都不能轻易篡改的证据层。具体而言，数据去重、study-aware 切分、grouped cross-validation、confirm 集、blind 集、指标定义、keep/revert 阈值和日志格式都必须在搜索开始前确定。agent 的自由只被允许存在于训练脚本、结构性建模假设以及有限的实验提案上。这样一来，系统才有可能真正优化“建模质量”，而不是优化“评估漏洞”。

换句话说，autoresearch 在生物信息学中的哲学转向是：把“自动寻找更高分模型”升级为“自动经营一个可靠的研究循环”。在这个循环中，agent 是高频探索者，确定性评估程序是裁判，人类研究者则掌握搜索空间设计权、证据标准定义权和最终结论解释权。

## 3. 设计原则
### 3.1 冻结证据层
生物信息学任务极易受到研究来源、患者来源、共享抗原、长度偏置、同源序列及标签构造规则的影响，因此评估协议本身必须先于模型搜索被冻结。冻结证据层意味着以下几个原则同时成立：数据预处理与去重不能被 agent 修改；confirm 与 blind 路由不能暴露给搜索代理；开发分数必须来自 grouped-CV 或其他显式控制泄漏的协议；保留或回滚候选改动的规则由确定性程序决定，而不是由 agent 自我解释。

### 3.2 搜索归纳偏置，而不是只搜索超参数
在许多生物信息学任务中，真正重要的问题是“模型是否显式表示了 mutant 与 wild-type 的差异”“是否把 HLA 条件交互写进了结构”“是否对 ranking 而不是 classification 进行了合适建模”。这些问题属于归纳偏置层，而不是普通超参数层。因此，适合 agent 的搜索空间应该优先包含特征块组织方式、显式对比头、HLA 条件分支、组内排序目标和局部结构路径，而不是先从学习率、dropout 或 label smoothing 开始。

### 3.3 盲测权和解释权必须保留给系统与人
agent 可以提出假设并改写训练代码，但不应直接触及 blind 集，更不应获得根据 blind 结果反向搜索的权限。否则所谓的“外部验证”会退化成另一个开发集。本文采用的理念是：agent 的探索终止于开发协议，confirm 和 blind 由冻结代码单独执行，人类研究者对最终结果进行解释和裁决。

### 3.4 每一轮实验都必须留下审计轨迹
一个可审计研究系统的最低要求是：每轮实验都应留下提案摘要、代码变更、运行日志、开发指标、keep/revert 决策以及失败类型。这样一来，负结果、失败设计和假进步都会成为研究知识的一部分，而不是被遗忘的隐性经验。

## 4. 系统方法
本文原型系统采用五个稳定入口。`prepare` 负责数据读取、特征构建、去重和 split manifest 写出；`train` 是 agent 唯一可修改的实验面；`confirm` 与 `blind_eval` 负责冻结外部评估；`controller` 负责实验状态机，包括准备数据、调用 agent、提交候选、运行训练、解析分数、执行 keep/revert 以及生成 smoke 报告。研究约束通过独立的 `program` 文档声明，以确保搜索策略能够在代码之外被单独维护。

系统采用“确定性控制器 + 受限 worker”结构。控制器负责准备输入上下文并调用 headless Codex worker，后者只能阅读研究约束和当前训练脚本，只能修改 `train.py`，且必须输出结构化提案。控制器随后运行训练、记录日志、比较当前分数与 champion 分数，并执行 keep 或 revert。若候选训练失败，则记录为 `train failed` 并自动回退到当前最优 commit。由此，失败轮不会打断整个研究过程，而会成为可审计负例。

当前原型中的开发目标分数定义为

\[
\mathrm{val\_score}=0.45\cdot \mathrm{AUPRC}+0.35\cdot \mathrm{PPV20}+0.10\cdot \mathrm{PPV10}+0.10\cdot \mathrm{NDCG20}.
\]

只有当候选分数相对于当前 champion 至少提升一个固定阈值时，该轮才会被保留；否则自动回滚。

## 5. 新抗原排序案例
### 5.1 任务定义
本文案例聚焦于 human HLA-I 条件下的 post-presentation neoantigen ranking。每个样本包含 mutant peptide、wild-type peptide、HLA 等位基因以及功能性标签。任务的目标不是单纯预测结合，而是在严格防泄漏设置下提高 top-k 排序质量。

### 5.2 特征与评估
原型系统使用 mutant 与 wild-type peptide 序列、HLA pseudosequence、NetMHCpan 预测、NetMHCstabpan 稳定性、BLAST-based foreignness、长度及理化特征等输入。随着系统迭代，标量特征被组织为显式的 feature blocks，包括基础块、comparison 块和 context 块。开发协议从早期的小规模单 fold smoke 升级到更大的 grouped-CV 开发集，并保留独立的 confirm 与 blind 评估。

### 5.3 搜索空间升级
最初的搜索主要集中于 sequence pooling、fusion gate 和局部 loss 变体。升级后，搜索空间被显式提升为三个层级。第一层是 feature block 组织方式，例如是否加入 WT-vs-Mut 差值块和上下文交互块。第二层是 contrast path，包括显式 WT-vs-Mut contrast head、HLA 条件 preference branch 以及交互摘要头。第三层是目标函数层，包括保持 BCE、引入 hybrid pairwise 或 pure pairwise 目标。通过这种组织，agent 被鼓励优先提出更接近生物学归纳偏置的改动，而不是重复进行低层调优。

## 6. 初步结果
### 6.1 评估协议本身会改变“什么是进步”
早期 smoke 过程中，许多改动在小型开发集和单 fold 设置下与 baseline 打平，看起来似乎有希望。但在开发集扩大并升级为 grouped-CV 之后，这些改动大多被证明只是伪进步，无法稳定超过更严格的 baseline。这个现象表明，在高偏倚任务中，强评估协议不是附属条件，而是搜索能否产生有效结论的前提。

### 6.2 高层 WT-vs-Mut 对比建模优于低层微调
在更大的 grouped-CV 开发协议下，最先带来真实提升的改动并非 pooling 或 loss 微调，而是高层的 HLA 条件 WT-vs-Mut 对比结构。基线模型的 grouped-CV 分数为 0.905841。在此基础上，引入显式 HLA 条件 WT-vs-Mut interaction delta 与标量条件对比调制后，分数提升到 0.909392。进一步引入共享的 `AllelePreferenceHead`，分别对 mutant-HLA 与 WT-HLA 配对进行建模，并将其 hidden delta 注入主 contrast path 后，开发分数提升到 0.932674。这一结果说明，在该任务上，更有效的研究方向不是继续堆叠低层微结构，而是显式编码 allele-aware mutant-versus-wild-type comparison。

### 6.3 pairwise ranking 不是默认赢家
尽管 ranking 任务在直觉上似乎应当更适合 pairwise 目标，但原型结果显示，直接将默认目标切换为轻量 hybrid pairwise 并未进一步提升性能，反而从 0.932674 降至 0.926356。这提示一个重要结论：在当前数据规模和结构下，pairwise objective 可能是一种有条件收益，而不是无条件收益。它需要依附于合适的表示层，而不能被当作自动更优的默认替代。

### 6.4 更多模块并不自动更好
增加 preference feature block、引入更复杂的 interaction summary head 或 preference-context head 并没有持续提高性能，有些改动明显掉分，有些则出现运行时失败。这个现象同样值得重视。它意味着在 autoresearch 设置下，系统真正寻找的不是“更复杂的架构”，而是“更贴近任务结构、同时足够稳定的归纳偏置”。复杂度本身并不是目标。

## 7. 讨论
### 7.1 对生物信息学的启示
本文案例提示，`autoresearch` 对生物信息学的价值不在于自动生成最终生物学结论，而在于迫使研究者把研究流程拆解为可冻结、可搜索和可验证的部分。数据清理和评估边界必须先被冻结，模型搜索才有意义；盲测权必须被系统保留，agent 搜索才不会污染证据；实验日志必须被持久记录，研究历史才具有知识价值。

### 7.2 对 agent 角色的重新定义
在这种系统中，agent 更像是“受约束的结构假设生成器”，而不是“自由科研主体”。它最擅长做的是在明确边界内高频提出新结构、改写代码并快速试错；它最不适合做的是定义外部证据标准、解释最终生物学结论或自由访问 blind 数据。换言之，agent 的最好位置并不是裁判席，而是实验台。

### 7.3 从 AutoML 到可审计研究系统
本文提出的核心概念转向是：把生物信息学中的自动化研究目标从“自动寻找最优超参数”升级为“自动经营一个可审计研究循环”。在这个循环里，真正被优化的不是单一模型，而是研究过程本身。这个视角可能比任何单轮性能提升更重要，因为它为日后扩展到其他任务提供了方法学基础。

## 8. 局限性
本文当前证据仍然存在明显局限。首先，现有结果来自一个本机原型与 smoke 级数据集，尚不足以支撑最终生物学结论或大规模 benchmark。其次，当前案例主要集中在新抗原排序，尚未在多个生物信息学任务上完成跨任务复现。第三，虽然系统已引入 grouped-CV、confirm 与 blind 协议，但尚未完成更大规模真实公开数据集上的公平预算对照，包括与随机搜索、贝叶斯优化、人工专家迭代以及更开放 agent 模式的直接比较。第四，当前搜索空间虽然已从低层微调提升到高层结构层，但对数据表征与模型复杂度的边界控制仍需进一步系统化。

## 9. 未来工作
下一阶段工作应优先围绕四个方向展开。第一，扩展到真实规模的数据集，并冻结更严格的 study-aware 和 time-aware 外部验证协议。第二，补做设计哲学层面的消融，包括是否使用 grouped-CV、是否保留 blind 隔离、是否限制 agent 只能修改训练脚本、是否优先搜索高层归纳偏置等。第三，将该系统迁移到第二个乃至第三个生物信息学任务上，以证明这不是新抗原排序的孤立工程技巧。第四，进一步完善实验日志、proposal schema 和运行时审计接口，使得每轮搜索不仅可回放，而且可供论文级图表自动生成。

## 10. 结论
本文提出，`autoresearch` 在生物信息学中的真正价值，不应被理解为“自动发现生物学规律”，而应被理解为“把研究过程本身重构为一个可冻结、可搜索、可盲测、可审计的研究系统”。在这一框架下，agent 负责高频探索，确定性评估代码负责裁决，人类研究者负责定义边界和解释证据。以新抗原排序为案例，我们展示了当开发协议足够严格、搜索面足够聚焦时，agent 不仅能够避免大量低收益微调，还能够发现更有效的 HLA 条件 WT-vs-Mut 对比建模结构。我们认为，这种从自动调参到可审计研究系统的转向，可能是 autoresearch 真正适合进入生物信息学的重要前提。
