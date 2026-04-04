# Strict-Confirm Run 11 Postmortem

## Scope

- Task: `neoantigen`
- Mode: `full`
- Budget: `20` rounds per strategy
- Policies:
  - `constrained + strict-confirm`
  - `random + strict-confirm`
- Result table:
  - `/Volumes/KINGSTON/tmp/autoneoag-strict-compare-r1/results.tsv`

## Main Outcome

The first budget-matched `strict-confirm` slice validated the gate itself, but did **not** show `constrained` beating `random`.

### Constrained

- `best_round = 1`
- `best_dev = 0.590922`
- `confirm = 0.387634`
- `blind = 0.527184`
- `keep_count = 1 / 20`

Important detail:

- Several rounds achieved much higher `dev` than the promoted champion.
- The highest discarded `dev` was `0.746480` at round `13`.
- Those rounds were rejected because their `confirm` scores stayed below the promoted champion.

Interpretation:

- The `strict-confirm` gate is doing real work.
- Under this policy, the current constrained loop is still too willing to exploit families that improve `dev` but do not survive `confirm`.

### Random

- `best_round = 5`
- `best_dev = 0.621568`
- `confirm = 0.395418`
- `blind = 0.559715`
- `keep_count = 2 / 20`

Interpretation:

- In this first slice, `random` was not obviously worse.
- It achieved a slightly stronger promoted champion than `constrained` on both `dev` and `confirm`.

## Why Constrained Underperformed Here

The main failure mode was not "the model families are useless". The failure mode was orchestration:

1. `strict-confirm` promoted only the baseline in the constrained run.
2. Once the promoted champion stayed at baseline, the frontier logic kept anchoring future proposals to the baseline branch.
3. The search therefore over-focused on the same exploit/recovery neighborhood, especially `gating` / `interaction_balance`.
4. Near-miss rounds that almost survived `confirm` were not promoted, but they also were not treated as meaningful frontier signals.

This produced a bad loop:

- `dev` spikes kept appearing
- `confirm` kept rejecting them
- the system still kept searching near those same discarded ideas

## Implication For The Search Runtime

The frontier cannot rely only on "promoted champion" in `strict-confirm` mode.

`strict-confirm` needs two parallel notions:

- `champion`
  - the official promoted winner
- `shadow_champion`
  - the strongest near-miss candidate by `confirm_round_score`

This lets the system remain conservative for promotion while still learning from discarded-but-promising candidates.

## Concrete Design Change

The next runtime version should be `strict-confirm-aware`:

1. Keep the official champion confirm-gated.
2. Track a shadow champion from discarded-but-confirm-checked rounds.
3. Promote families in the frontier using:
   - `best_confirm_score`
   - `confirm_survival_rate`
   - `near_miss_count`
4. Avoid freezing a family too early if it repeatedly produces near-miss confirms.

## Expected Benefit

This change should reduce the current failure mode:

- "baseline remains champion forever, so constrained search keeps orbiting around low-value exploit neighborhoods"

Instead, the loop should become:

- "official keep stays conservative, but the next prompt is guided by the best confirm-adjacent direction rather than only by the last promoted baseline"
