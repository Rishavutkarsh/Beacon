# Beacon Candidate Selection Report

## Inputs
- Dev split only: 95 prompts.
- Candidates: base, checkpoint-175, checkpoint-400, checkpoint-448.
- Final eval was not used.
- Two blind LLM judges reviewed every row:
  - Judge A: safety/source-boundary.
  - Judge B: usefulness/task-fidelity.

## Result
- Base won both judges after decoding blind labels.
- Among SFT adapters, checkpoint-175 was strongest.
- Checkpoint-400 and checkpoint-448 were worse than checkpoint-175 on average score, first-place votes, major issues, and generic-template flags.

## Aggregate Scores
| Candidate | Avg score | First-place votes | Critical flags | Major flags | Generic flags | Avg chars |
|---|---:|---:|---:|---:|---:|---:|
| base | 3.000 | 106 | 21 | 74 | 4 | 719.8 |
| checkpoint-175 | 2.516 | 42 | 13 | 96 | 33 | 462.0 |
| checkpoint-400 | 2.105 | 27 | 20 | 120 | 65 | 477.9 |
| checkpoint-448 | 2.068 | 15 | 16 | 120 | 73 | 472.4 |

## Recommendation
Do not select checkpoint-400 or checkpoint-448 based on this dev judge pass. If choosing an SFT adapter anyway, checkpoint-175 is the least-bad adapter, but the stronger recommendation is to pause promotion and inspect the adapter failure mode before running final_eval.

The pattern is consistent with overtraining/template drift: train loss fell hard, dev loss bottomed before one epoch, and later checkpoints became more generic without behavioral gains.
