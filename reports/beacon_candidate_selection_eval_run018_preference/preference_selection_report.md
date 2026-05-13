# Beacon Preference Judge Rerun Report

## Inputs
- Reused existing dev generations only: base, checkpoint-175, checkpoint-400, checkpoint-448.
- Final eval remained untouched.
- Judges used compact expected-behavior notes instead of the full target as the main anchor.
- Two blinded judges reviewed all 95 dev rows and 20 label-shuffle stability rows.
- One decoded sample reviewer inspected 26 stratified rows.

## Aggregate Result
The veto-first rule selects **checkpoint-175** as the best adapter candidate, but not as a promotion-ready model.

| Candidate | Best after veto | Runner-up | Safety veto | Unsafe/off-target |
|---|---:|---:|---:|---:|
| base | 99.0 | 23.0 | 33 | 60 |
| checkpoint-175 | 38.0 | 73.0 | 15 | 90 |
| checkpoint-400 | 27.0 | 49.5 | 24 | 116 |
| checkpoint-448 | 26.0 | 40.5 | 20 | 123 |

Interpretation:
- Base still wins many preference votes, mostly because it is more complete and fluent.
- Base also gets the most safety vetoes, so it is not a clean winner under a crisis-safety selection rule.
- Among adapters, checkpoint-175 is strongest and has the fewest safety vetoes.
- Checkpoint-400 and checkpoint-448 look worse than checkpoint-175 and should not be selected.

## Sample Review
The decoded sample reviewer found the aggregate only partially trustworthy:
- preferred counts in the 26-row sample were close: base 11, checkpoint-175 10, checkpoint-400 7, checkpoint-448 6.
- unsafe counts were also close: checkpoint-400 12, checkpoint-175 13, base 14, checkpoint-448 15.
- reviewer's forced provisional adapter choice was checkpoint-175, but with caution.

The sample reviewer specifically warned that no candidate is clean without targeted investigation of:
- food/flood contamination,
- baby formula or vulnerable-person water,
- generator/CO boundaries,
- medical/photo uncertainty.

## Recommendation
Use **checkpoint-175 only as the provisional best adapter for further behavioral diagnosis**, not as a final/promotion candidate.

Do **not** select checkpoint-400 or checkpoint-448. Do **not** run final_eval yet unless the goal is a diagnostic comparison rather than candidate promotion.
