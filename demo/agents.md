# demo/agents.md

Scope: UI demo (local) for tri-state prediction.

Requirements:
- Accept image upload
- Load:
  - trained checkpoint
  - temperature.json (optional)
  - conformal.json (optional)
- Output:
  - AI / Real / Abstain
  - calibrated confidence (max prob after temperature scaling)
  - prediction set Γ(x) and whether abstained
  - optionally a residual heatmap if DIRE branch is enabled and available

Keep demo isolated:
- no training code inside demo
- must run from repo root
