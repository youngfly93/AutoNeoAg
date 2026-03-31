# Manifests

本目录记录 AutoNeoAg framework 的 **Phase 1 evidence-layer drafts**。

当前这些 manifest 还不是 full ingest 结束后的最终版本，而是：

- 先冻结任务边界
- 先冻结 source 级纳入 / 排除规则
- 先冻结 split / lockbox 逻辑
- 为后续 `prepare.py --mode full` 提供明确输入

当前阶段采用两层定义：

1. **source-level manifest**
   用于说明哪些数据源、哪些年份、哪些 study 类别允许进入 train / confirm / blind

2. **sample-level manifest**
   在 Phase 2 / 3 full ingest 后生成

当前目录先提供 source-level draft：

- `neoantigen/`
- `hla_immunogenicity/`

这些文件的目标不是替代最终数据整理，而是确保 full ingest 开始前，证据层规则已经先被写下来并可审阅。
