# src/owaid/calibration/agents.md

Scope: temperature scaling + split conformal prediction for classification.

Temperature scaling:
- Fit scalar T > 0 by minimizing NLL on calibration set
- Save artifact:
  outputs/runs/<run_id>/calibration/temperature.json
  { "temperature": float, "n": int, "nll_before": float, "nll_after": float }

Conformal classification:
- Use calibrated probs p_T(y|x)
- Nonconformity s(x,y)=1-p_T(y|x)
- Calibration scores computed on calibration set using TRUE labels
- Quantile selection must follow split conformal convention:
  qhat = quantile_{ceil((n+1)*(1-alpha))/n}(scores)
- Save artifact:
  outputs/runs/<run_id>/calibration/conformal.json
  {
    "alpha": float,
    "qhat": float,
    "method": "split" | "mondrian",
    "class_qhat": { "0": float, "1": float } (if mondrian),
    "n": int
  }

Tri-state decision is implemented in src/owaid/models/abstention.py (not here).
