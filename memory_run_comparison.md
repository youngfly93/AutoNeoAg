# Neoantigen Run Comparison

This note compares the three full `neoantigen` constrained runs we have finished so far:

- `run_01`: original full constrained search without research-memory frontier guidance
- `run_02`: first memory-aware full run
- `run_03`: second memory-aware full run

## Summary

| Run | Mode | Best Round | Best dev | Confirm | Blind | dev-blind gap | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| `run_01` | baseline constrained | 97 | 0.784791 | 0.274226 | 0.385888 | 0.398903 | Highest `dev`, weakest external generalization |
| `run_02` | memory-aware | 62 | 0.756344 | 0.404561 | 0.669962 | 0.086382 | Best `blind`, strongest generalization-oriented run |
| `run_03` | memory-aware | 48 | 0.778706 | 0.318851 | 0.605542 | 0.173164 | Best balance so far: near-`run_01` dev with much stronger blind |

## Key Takeaways

1. `run_01` is still the highest `dev` peak, but it overfits most strongly to the development protocol.
2. `run_02` shows the clearest evidence that research-memory guidance improves blind generalization.
3. `run_03` is the most balanced trajectory:
   - `dev` is close to `run_01`
   - `blind` is much better than `run_01`
   - search efficiency is better than both prior runs via a higher keep rate
4. Taken together, the current evidence supports a directional conclusion:
   - research-memory-aware search does **not always maximize the absolute best `dev`**
   - but it **does appear to improve search efficiency and reduce the dev-to-blind collapse**

## Keep Rate

| Run | Rounds | Keeps | Keep Rate |
|---|---:|---:|---:|
| `run_01` | 100 | 6 | 6.0% |
| `run_02` | 100 | 9 | 9.0% |
| `run_03` | 100 | 11 | 11.0% |

This is one of the strongest positive signals for the memory-aware controller. Even before discussing external metrics, the controller is wasting fewer rounds.

## Run 01

task_id: `neoantigen`  
strategy: `constrained`  
run_id: `1`  
best_round: `97`  
best_commit: `dec05e1`  
best_val_score: `0.784791`

### Confirm

```json
{
  "auprc": 0.2235765287881055,
  "ndcg20": 0.38616484737745765,
  "ppv10": 0.3,
  "ppv20": 0.3,
  "split": "confirm",
  "val_score": 0.27422592269239326
}
```

### Blind

```json
{
  "auprc": 0.4471833263477782,
  "ndcg20": 0.471550912281904,
  "ppv10": 0.5,
  "ppv20": 0.25,
  "split": "blind",
  "val_score": 0.38588758808469054
}
```

## Run 02

task_id: `neoantigen`  
strategy: `constrained`  
run_id: `2`  
best_round: `62`  
best_commit: `056e096`  
best_val_score: `0.756344`

### Confirm

```json
{
  "auprc": 0.4396008803462923,
  "ndcg20": 0.617409085414645,
  "ppv10": 0.4,
  "ppv20": 0.3,
  "split": "confirm",
  "val_score": 0.4045613046972961
}
```

### Blind

```json
{
  "auprc": 0.753590277503321,
  "ndcg20": 0.7584676478583555,
  "ppv10": 0.8,
  "ppv20": 0.5,
  "split": "blind",
  "val_score": 0.66996238966233
}
```

## Run 03

task_id: `neoantigen`  
strategy: `constrained`  
run_id: `3`  
best_round: `48`  
best_commit: `9413aa1`  
best_val_score: `0.778706`

### Confirm

```json
{
  "auprc": 0.3776507563536334,
  "ndcg20": 0.48908219498851113,
  "ppv10": 0.3,
  "ppv20": 0.2,
  "split": "confirm",
  "val_score": 0.31885105985798606
}
```

### Blind

```json
{
  "auprc": 0.6844758097110013,
  "ndcg20": 0.70027533044357,
  "ppv10": 0.7,
  "ppv20": 0.45,
  "split": "blind",
  "val_score": 0.6055416474143076
}
```

## Figure

Comparison figure generated locally:

- `artifacts/figures/neoantigen_run_comparison.png`

The figure is intentionally kept out of git because `artifacts/` is ignored in this repository. The generating script is versioned:

- [`scripts/plot_run_comparison.py`](scripts/plot_run_comparison.py)
