# src/owaid/models/agents.md

Scope: CLIP baseline detector + optional residual/DIRE fusion + abstention wrappers.

Baseline model requirements:
- Uses open_clip
- Encoder frozen by default
- forward(images) returns dict:
  - logits: torch.FloatTensor [B,2]
  - probs: torch.FloatTensor [B,2] (softmax over logits, optional for training)
  - features: torch.FloatTensor [B,D] (CLIP embedding) when requested

Config knobs:
- backbone.model_name, backbone.pretrained
- backbone.freeze (bool)
- backbone.unfreeze_last_n (int, optional)
- head.hidden_dims, head.dropout

Fusion/DIRE branch:
- Must be optional and behind cfg.model.use_dire
- Residual compute must support "offline cache" mode to avoid recomputation
- Fusion default: concat([clip_emb, residual_emb]) -> MLP -> logits

Abstention:
- abstention policy must not change trained classifier weights
- uses calibration artifacts: temperature.json and conformal.json
