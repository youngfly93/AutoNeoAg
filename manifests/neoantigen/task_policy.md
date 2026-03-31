# Neoantigen Task Policy

## Role In The Paper

This is the primary case-study task for the framework paper.

## Hard Constraints

- never train on TESLA
- never expose TESLA labels to the agent
- never merge 2024-2025 holdout sources into the main training pool
- never treat simulated labels as equal to functional human labels
- never accept a gain that is driven only by length or source bias

## Priority Baselines

- presentation-only baseline
- logistic regression on scalar features
- random forest
- xgboost
- lightgbm
- fusion baseline from current `train.py`

## Priority Search Directions

- WT-vs-Mut contrast structure
- HLA-conditioned interaction structure
- grouped ranking objective only if representation change is paired with it
- auxiliary use of weak labels only after strict/main pipeline is stable

## Suspicious Gains

- gains that vanish on confirm or blind
- gains that only improve a single study family
- gains that correlate mostly with length buckets
- gains that appear only after introducing binding-derived weak-label shortcuts

## Promotion Rule To Phase 2

Task A can enter full ingest implementation only after:

- source manifest is accepted
- lockbox manifest is accepted
- split manifest is accepted
- full source list for first-pass adapters is selected
