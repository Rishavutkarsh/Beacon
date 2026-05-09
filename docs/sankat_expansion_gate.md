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

Phase 2 `v2_1015` uses the expanded reviewed seed bank:

```bash
python scripts/build_expansion.py --profile v2_1015 --stage full \
  --seed-cards data/seed_cards/sankat_saathi_seed_cards_v2_train_expanded.jsonl \
  --rule-manifest data/seed_cards/source_rule_manifest_v2_train_seed_expansion.jsonl \
  --out-dir data/expanded/beacon_v2_1015/run_001 \
  --fail-if-exists
python scripts/validate_expansion.py --profile v2_1015 \
  --rule-manifest data/seed_cards/source_rule_manifest_v2_train_seed_expansion.jsonl \
  --run-dir data/expanded/beacon_v2_1015/run_001
python scripts/make_audit_bundle.py --run-dir data/expanded/beacon_v2_1015/run_001
```

The profile asserts the seed snapshot has 203 train, 40 dev, and 40 final-eval
seeds. It targets exact accepted counts of 1015 train, 120 dev, and 120
final-eval rows. Final-eval is treated as a strict isolated split: exact
final-eval examples must not feed train/dev repair or tuning prompts.
Each run must use an immutable directory such as `run_001`, `run_002`, and
`run_003`; do not rerun generation into a prior non-empty run directory.

## Required Approval Artifacts

Every validation run writes:

- `dataset_manifest.json`
- `schema_validation_report.json`
- `lineage_validation_report.json`
- `split_leakage_report.json`
- `source_grounding_report.csv`
- `source_claim_support_report.csv`
- `safety_lint_report.json`
- `output_similarity_report.csv`
- `pattern_collapse_report.json`
- `per_seed_diversity_report.json`
- `final_eval_isolation_report.json`
- `quota_report.json`
- `behavior_distribution_report.json`
- `deterministic_gate_report.json`
- `review_sampling_manifest.json`
- `commands_transcript.jsonl`
- `environment_manifest.json`
- `git_manifest.json`
- `input_snapshot_manifest.json`
- `critic_report.jsonl`
- `subagent_review_report.jsonl`
- `reviewer_decisions.jsonl`
- `repair_lineage.jsonl`
- `repair_prompt_lineage.jsonl`
- `row_failure_ledger.jsonl`
- `review_calibration_report.json`
- `accepted_rows.jsonl`
- `final_accepted_rows.jsonl`
- `rejected_rows.jsonl`
- `rejected_row_ledger.jsonl`
- `dataset_freeze_manifest.json`
- `freeze_decision.md`
- `run_summary.md`

`critic_report.jsonl` and `subagent_review_report.jsonl` are placeholder
records until live calibrated critic/subagent passes are run. Placeholder
reviews deliberately fail the approval gate.

`dataset_freeze_manifest.json` is the only training approval signal. Training
must not start unless it reports `status: pass` and hashes the seed snapshot,
rule manifest, generated candidates, final accepted rows, rejected ledger, and
review artifacts.

When a run fails, use `row_failure_ledger.jsonl` to decide whether the next
immutable run repairs renderer output, source grounding, leakage, pattern
diversity, or reviewer artifacts. Any train/dev repair must write
`repair_prompt_lineage.jsonl` and prove exact final-eval text was not used.
