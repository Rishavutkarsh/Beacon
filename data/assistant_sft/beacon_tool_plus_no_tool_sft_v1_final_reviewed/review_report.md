# Beacon Tool + No-Tool SFT v1 Reviewed Mix

This package combines main-reviewed tool-aware rows with a reviewed slice of the new no-tool natural SFT candidate rows.

- Total rows: 1390
- Tool-call/evidence rows: 950
- No-tool rows total: 440
- No-tool rows added from natural candidate: 200
- Splits: {'train': 1126, 'dev': 132, 'final_eval': 132}

No training has been launched. `training_export_allowed` remains false for final inspection.

Main no-tool edits applied:
- Rewrote the ORS/tablet stock row with a cut foot so it no longer uses a medicine-dose request template.
- Normalized the clinic newborn ward prompt to avoid teaching 'tool' language in a no-tool row.

Known residual risk:
- The full-trace tool-aware source was missing 10 row IDs from the 1152 reviewed packet decisions, so those 10 are not included here.
- Packet 05 shelter-conflict rows are intentionally downsampled to 15 because the packet still has repeated patterns.
