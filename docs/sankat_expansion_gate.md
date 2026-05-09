# Sankat Saathi Expansion Gate

This gate turns the reviewed 200 seed cards into auditable expansion candidates.
It is intentionally conservative: generated rows are not training-ready until all
deterministic gates pass and calibrated subagent review artifacts replace the
placeholder review records.

## Commands

Calibration pilot:

```bash
python scripts/build_expansion.py --profile calibration
python scripts/validate_expansion.py --profile calibration --skip-final-count-gate
python scripts/make_audit_bundle.py --run-dir data/expanded/sankat_expansion_calibration
```

Phase 1 `v1_600` candidate:

```bash
python scripts/build_expansion.py --profile v1_600
python scripts/validate_expansion.py --profile v1_600
python scripts/make_audit_bundle.py --run-dir data/expanded/sankat_expansion_v1_600
```

Phase 1 uses the current reviewed 200 seeds: 120 train seeds at up to 5 variants
for 600 train rows, plus 40 dev and 40 final-eval seeds at 3 variants each for
120 dev and 120 final-eval rows. Dev/final rows remain held out from training.

Phase 2 `v2_1k` remains infeasible until more reviewed train-only seeds are
added. With the existing 120 train seeds and max-5 cap, train capacity is still
600, so `v2_1k` correctly exits nonzero unless 60-80 additional train seeds are
merged into the seed bank.

## Required Approval Artifacts

Every validation run writes:

- `dataset_manifest.json`
- `schema_validation_report.json`
- `split_leakage_report.json`
- `source_grounding_report.csv`
- `safety_lint_report.json`
- `pattern_collapse_report.json`
- `quota_report.json`
- `critic_report.jsonl`
- `subagent_review_report.jsonl`
- `repair_lineage.jsonl`
- `accepted_rows.jsonl`
- `rejected_rows.jsonl`
- `run_summary.md`

`critic_report.jsonl` and `subagent_review_report.jsonl` are placeholder
records until live calibrated critic/subagent passes are run. Placeholder
reviews deliberately fail the approval gate.
