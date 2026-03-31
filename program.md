# AutoNeoAg Program

Goal:
Maximize grouped-validation ranking quality on STRICT human HLA-I data.
Primary objective is PPV20/AUPRC, not raw AUROC.

Hard constraints:
- Never touch blind or lockbox data from the Codex worker.
- Only `train.py` may be modified by the Codex worker.
- Gains driven only by peptide length are suspicious.
- Simpler models win ties.
- No feature may leak label construction rules.
- Training code must remain MPS/CPU compatible.
- Avoid isolated tweaks to loss shape, label smoothing, class weighting, sampling, or output-bias priors unless paired with a structural representation change.

Priority directions:
1. Use more of the existing strict-label feature set, especially WT-vs-Mut comparisons, stability, foreignness, and delta-derived signals.
2. Strengthen sequence encoder and scalar + sequence fusion on strict labels.
3. Preserve WT-vs-Mut delta information and HLA-conditioned interactions.
4. Prefer stable representation changes over micro-tuning the objective.

Required reporting:
- `val_score`
- `auprc`
- `ppv10`
- `ppv20`
- `ndcg20`
- `peak_memory_mb`
- `num_params`
