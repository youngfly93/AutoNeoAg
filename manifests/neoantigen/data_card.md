# Neoantigen Data Card

## Status

Phase 1 draft. This file freezes task intent and evidence-layer boundaries before full ingest is implemented.

## Task Definition

- Task ID: `neoantigen`
- Task type: `human HLA-I post-presentation top-k ranking`
- Unit of prediction: `mutant peptide + wild-type peptide + HLA`
- Primary outcome: `functional immunogenicity ranking`

## In-Scope Samples

- human samples only
- HLA-I only
- peptide length `8-11`
- mutant / wild-type paired records
- records with functional evidence preferred

## Out-of-Scope Samples

- mouse-only validation
- simulated positives / simulated negatives
- binding-only labels without functional context
- presentation-only labels used as direct task labels
- mixed-class tasks that do not preserve ranking semantics

## Label Policy

### Tier A

- human
- HLA-I
- functional positive / functional negative
- eligible for main training and model selection

### Tier B

- human
- curation quality is acceptable but functional context is weaker
- eligible for auxiliary or lower-weight training only

### Tier C

- simulated
- weak labels
- mouse labels
- not eligible for main conclusion

## Lockbox Policy

Two lockboxes are fixed at the source-policy level:

1. `tesla_lockbox`
2. `timesplit_2024plus_lockbox`

These lockboxes are blind to the agent. They are not development sets.

## Confirm Policy

- confirm must come from training-eligible source families
- confirm must not reuse the same study grouping as dev
- confirm is for human review and sanity confirmation, not for continuous search

## Blind Policy

- blind is source-separated and time-separated
- any source explicitly assigned to lockbox is excluded from train and confirm

## Known Shortcut Risks

- peptide length bias
- HLA frequency bias
- study / assay bias
- label-construction leakage from binding-derived rules
- repeated mutation-event leakage across folds

## Full Ingest Entry Criteria

Full ingest for this task can start once:

- source manifest is reviewed
- lockbox manifest is reviewed
- split manifest is reviewed
- raw source list is complete enough to implement adapters
