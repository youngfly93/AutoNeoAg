# HLA Immunogenicity Data Card

## Status

Phase 1 draft. This file freezes task scope and evidence-layer boundaries before full ingest is implemented.

## Task Definition

- Task ID: `hla_immunogenicity`
- Task type: `human HLA-I peptide immunogenicity ranking`
- Unit of prediction: `mutant-like peptide + paired reference peptide + HLA`
- Primary outcome: `functional T-cell immunogenicity ranking`

## In-Scope Samples

- human samples only
- HLA-I only
- peptide length `8-11`
- peptide / HLA pairs with functional immunogenicity evidence
- studies with enough provenance to support grouped splitting

## Out-of-Scope Samples

- mouse-only datasets
- pure binding datasets without downstream functional assay
- presentation-only datasets used as direct task labels
- synthetic negatives whose construction leaks the label rule
- mixed-class pTCR tasks that are not comparable to peptide ranking

## Label Policy

### Tier A

- human
- HLA-I
- functional immune assay support
- eligible for main training and model selection

### Tier B

- human
- weaker curation or incomplete assay context
- eligible only as lower-weight training support

### Tier C

- simulated
- mouse-only
- weak labels without functional endpoint
- excluded from the main conclusion

## Confirm Policy

- confirm remains inside training-eligible source families
- confirm should be grouped away from dev studies when feasible

## Blind Policy

Blind policy for this task is source-oriented rather than benchmark-oriented:

1. `study_external_lockbox`
2. `timesplit_2024plus_lockbox`

## Known Shortcut Risks

- public immunogenicity benchmark overlap
- HLA frequency imbalance
- length bias
- assay-specific bias
- negatives constructed from binding thresholds alone

## Full Ingest Entry Criteria

Full ingest for this task can start once:

- source manifest is reviewed
- lockbox manifest is reviewed
- split manifest is reviewed
- first-pass source adapters are scoped
