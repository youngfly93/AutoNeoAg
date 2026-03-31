# HLA Immunogenicity Task Policy

## Role In The Paper

This is the second full task used to demonstrate near-neighbor transfer of the framework.

## Hard Constraints

- never let external study holdout become a development set
- never let 2024-2025 holdout become a development set
- never promote binding-only labels to the main endpoint
- never mix mouse-only evidence into the main human claim
- never accept a gain that disappears once grouped by study or HLA family

## Priority Baselines

- logistic regression on scalar features
- random forest
- xgboost
- lightgbm
- small MLP
- current fusion baseline from `train.py`

## Priority Search Directions

- HLA-conditioned comparison structure
- sequence/context fusion that transfers from Task A cleanly
- grouped ranking objective only when paired with representation change
- stability of gains across study-aware splits

## Suspicious Gains

- gains that only appear on single studies
- gains that mostly track HLA frequency
- gains that depend on binding-only shortcut signals
- gains that collapse on confirm or blind

## Promotion Rule To Phase 3

Task B can enter full ingest implementation only after:

- source manifest is accepted
- lockbox manifest is accepted
- split manifest is accepted
- full source list for first-pass adapters is selected
