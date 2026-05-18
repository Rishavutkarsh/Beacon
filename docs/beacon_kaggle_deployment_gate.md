# Beacon Kaggle Deployment Gate

This is the standing rule for Beacon Kaggle training and eval pushes.

Before any material Kaggle deployment, run a quick code review gate:

1. Local sanity check
   - Confirm the intended notebook/script and metadata paths.
   - Run a syntax check for Python scripts.
   - Verify dataset, model, adapter, and output paths.

2. Training config check
   - Record max sequence, prompt, and completion lengths.
   - Record batch size, gradient accumulation, eval cadence, save cadence, and max steps.
   - Check the config against the target GPU. For T4 DPO, assume memory is tight because chosen/rejected pairs multiply sequence work, and reference-model paths can double forward memory.

3. Reviewer check
   - Get at least one quick independent code-review pass before pushing material training or eval changes.
   - For high-risk changes, get two reviewers when time allows: one focused on correctness and one focused on memory/runtime fit.
   - Do not push if a reviewer flags a blocker.

4. Deployment check
   - Push only after local sanity and reviewer gates pass.
   - After push, check Kaggle status quickly and pull logs immediately on failure.
   - Do not wait a long runtime window for failures that should be caught by import, path, or first-step memory checks.

Hotfixes can use a fast version of this gate, but the gate still applies.
