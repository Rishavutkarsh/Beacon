# Train-Only Seed Expansion Workflow

This workflow generates additional Sankat Saathi train-only seed cards while
protecting locked dev/final seeds from semantic leakage.

## Create Contained Packets

```bash
python scripts/create_seed_expansion_packets.py
```

Outputs are written to `data/seed_cards/expansion_v2_train_only/`:

- `generator_packet.json`: train seeds, redacted protected-cluster summaries,
  target gaps, and required proposal fields. This is the packet for generation
  subagents.
- `reviewer_gate_packet.json`: full 200 seed cards plus v1_600 leakage/template
  reports. This is only for reviewers and gates.
- `subagent_assignments.json`: distinct generation slices, 15-20 proposals each.

Generator subagents should not receive exact dev/final cards.

## Validate Proposal Batches

Each proposal JSONL must use canonical seed-card fields plus the extra
anti-leakage fields listed in `generator_packet.json`.

```bash
python scripts/validate_seed_proposals.py path/to/proposals.jsonl
```

The gate writes:

- `accepted_seed_proposals.jsonl`
- `review_seed_proposals.jsonl`
- `rejected_seed_proposals.jsonl`
- `seed_proposal_neighbors.csv`
- `seed_metadata_collision_report.csv`
- `seed_proposal_gate_report.json`

The command exits nonzero until at least 80 proposals pass with no hard
violations. Borderline rows go to review or reject; do not accept marginal cards
to hit a quota.

## Hard Principles

- Dev/final are protected exclusion zones.
- Train additions must be different decision problems, not paraphrases.
- Source grounding beats novelty.
- Generic fictional local details are allowed; invented operational facts are not.
- v1_600 generated rows are used only as a negative anti-template audit, not as
  generation material.
