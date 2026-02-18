# src/owaid/metrics/agents.md

Scope: numeric metrics + plot-ready curve data.

Must implement:
- auroc(y_true, y_score)
- tpr_at_fpr(y_true, y_score, fpr=0.01)
- ece(probs, labels, n_bins=15) + reliability curve bins
- risk_coverage(confidence, correct_mask) -> curve points + AURC
- abstention metrics:
  abstain_rate, selective_accuracy, coverage

Conventions:
- Binary labels: 0=Real, 1=AI
- y_score for AUROC is score for class 1 (AI), shape [N]
- probs shape [N,2]
- risk = 1-accuracy on answered subset
