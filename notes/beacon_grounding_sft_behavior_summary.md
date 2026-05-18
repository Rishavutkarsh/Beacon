# Beacon Behavior Summary For Grounding-Aware SFT

## Current Best Signals

- CPT `checkpoint-300` remains the best pure domain-knowledge checkpoint on the 60-row heldout MCQ headline metric.
- Earlier SFT `sft_v1_ckpt300_best` preserves most CPT knowledge and safety better than the later 2-epoch SFT final.
- Later SFT `sft_v2_final` improves source-QA partial usefulness versus base, but adds many unsupported factual extras.

## MCQ Knowledge Results

| Model | Heldout MCQ | Overall | Critical safety | Unsafe picks |
|---|---:|---:|---:|---:|
| Base Gemma | 49/60 = 81.67% | 85.00% | 43/54 = 79.63% | 3 |
| CPT ckpt300 | 53/60 = 88.33% | 86.25% | 45/54 = 83.33% | 1 |
| SFT v1 ckpt300 best | 52/60 = 86.67% | 87.50% | 45/54 = 83.33% | 1 |
| SFT v2 final | 51/60 = 85.00% | 85.00% | 43/54 = 79.63% | 2 |

## Source-QA Judge Results For SFT v2 Final

| System | Strict correct | Partial or correct | Avg score | Unsupported extras |
|---|---:|---:|---:|---:|
| Base Gemma | 5/59 = 8.47% | 23.73% | 0.322 | 14 |
| SFT v2 final | 5/58 = 8.62% | 39.66% | 0.483 | 52 |

Paired source-QA: SFT v2 final beats base on 16 rows, loses on 10, ties on 31. It is more helpful/complete, but much more likely to make unsupported source-specific claims.

## Failure Pattern

- Exact safety constants are fragile: `40 F`, `4 hours`, `24 hours`, `1 minute`, `3 minutes`, `30 minutes`, `15g quick carbs`.
- Later SFT can produce longer, more confident answers without enough evidence discipline.
- MCQ free generation exposed instruction-echo behavior; forced-choice logprob scoring fixed the eval protocol, but the echo is still a behavioral warning.
- The SFT dataset teaches crisis-answer style, but not a strong "retrieve/check exact facts before answering" habit.

## Grounding-Aware SFT Design Implications

- Do not rely on SFT to memorize all numeric safety facts.
- Add an offline fact-card/RAG step for source-specific constants and policy facts.
- Train the assistant behavior to say: check local fact card first for exact numbers; answer only from retrieved card when the question asks for thresholds, timings, dosages, temperatures, or official roles.
- Include negative examples where unsupported details are penalized even if the answer is fluent and safety-sounding.
- Keep CPT ckpt300 as the knowledge base; use SFT only if it preserves safety and improves grounded behavior.
