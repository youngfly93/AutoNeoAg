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

Priority directions:
1. Strengthen scalar + sequence fusion on strict labels.
2. Explore focal/ranking losses before architecture bloat.
3. Preserve WT-vs-Mut delta information.
4. Prefer stable improvements over large but noisy jumps.

Required reporting:
- `val_score`
- `auprc`
- `ppv10`
- `ppv20`
- `ndcg20`
- `peak_memory_mb`
- `num_params`

