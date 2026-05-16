# Beacon Source-QA Eval Notes

- Base-only Qwen judge was useful and completed first.
- Paired base-vs-CPT judge was launched from the original comparison script after CPT generation completed.
- This rejudges base unnecessarily. For future evals, prefer separate judge runs per candidate, then aggregate local result files afterward.
- Current paired run should be allowed to finish because it is already running and will provide the CPT comparison signal.
