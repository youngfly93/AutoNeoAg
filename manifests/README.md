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

## Source Manifest 字段约定

`source_manifest.csv` 当前采用统一字段：

- `source_id`
- `source_name`
- `task_id`
- `source_type`
- `download_method`
- `adapter_id`
- `ingest_status`
- `license_or_access`
- `raw_file_path`
- `source_priority`
- `expected_format`
- `assay_scope`
- `species_scope`
- `hla_scope`
- `split_role`
- `label_strength`
- `is_train_eligible`
- `is_confirm_eligible`
- `is_blind_only`
- `year_start`
- `year_end`
- `normalization_profile`
- `notes`

关键字段解释：

- `adapter_id`
  对应后续 full ingest 的 source adapter 标识。

- `ingest_status`
  当前允许的值：
  - `planned`
  - `manual_required`
  - `implemented`
  - `external_lockbox`

- `source_priority`
  用于确定 full ingest 的首批接入顺序。数值越小越优先。

- `expected_format`
  预期原始文件格式，例如 `csv`, `tsv`, `xlsx`, `json`, `mixed_manual_bundle`。

- `split_role`
  当前允许的值：
  - `train_candidate`
  - `confirm_candidate`
  - `blind_only`
  - `excluded_aux_only`

- `normalization_profile`
  说明该来源进入标准化表前需要套用的字段映射 / 清洗规则。

## Lockbox Manifest 说明

`lockbox_manifest.csv` 当前是 source-level selector，不是最终 sample-level lockbox。

Phase 2 / 3 full ingest 后，会将 source-level selector 扩展为：

- source-level selector
- study-level selector
- sample-level selector

当前阶段只要求先把 blind / exclusion 边界写死，不要求先列出全部样本。

## 当前可用的 full ingest 辅助命令

已实现 source adapter 的任务，现在可以直接用下面两类命令：

1. 生成 raw 模板：

```bash
python scripts/bootstrap_full_raw_templates.py --task neoantigen
python scripts/bootstrap_full_raw_templates.py --task hla_immunogenicity
```

2. 校验单个 source 是否能被标准化：

```bash
python scripts/validate_full_source.py --task neoantigen --source neo_iedb_functional
python scripts/validate_full_source.py --task neoantigen --source neo_literature_curated
python scripts/validate_full_source.py --task hla_immunogenicity --source immuno_iedb_functional
```

3. 运行 task-level full prepare planning：

```bash
python prepare.py --task neoantigen --mode full
python prepare.py --task hla_immunogenicity --mode full
```

在当前阶段，`prepare.py --mode full` 会：

- 读取并校验 manifests
- 解析 source adapter 状态
- 对已实现且有 raw 文件的 source 执行标准化
- 输出 `full_prepare_plan.json`
- 产出 task-level `dataset.parquet`
- 产出 sample-level `splits_grouped_v1.json`
- 产出 `source_index.tsv`

它还不会直接声明 full ingest 完成；真正 full-ready 仍然依赖更多 source adapter 和真实 raw 数据补齐。
