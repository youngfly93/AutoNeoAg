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
- Avoid isolated tweaks to loss shape, label smoothing, class weighting, sampling, or output-bias priors unless paired with a higher-level representation change.

Priority directions:
1. Add or reorganize scalar feature blocks, especially WT-vs-Mut comparison blocks, stability, foreignness, and delta-derived context.
2. Strengthen the explicit WT-vs-Mut contrast path before touching local pooling details.
3. Explore pair/group ranking objectives only when paired with a contrast head or new feature block, not as a standalone tweak.
4. Preserve WT-vs-Mut delta information and HLA-conditioned interactions.
5. Prefer stable high-level changes over repeated pooling/fusion micro-variants.

Required reporting:
- `val_score`
- `auprc`
- `ppv10`
- `ppv20`
- `ndcg20`
- `peak_memory_mb`
- `num_params`
