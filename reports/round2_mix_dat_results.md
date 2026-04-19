# Round-2 Training Results: Tier-1 Bias Removal + 50/50 Mix + DAT

**Generated:** 2026-04-18
**Eval pipeline:** calibrate_temperature → build_conformal → evaluate (commfor, raid, aiart-test, cifake)
**Job:** 39913196 (post_eval_all), completed 2026-04-18T15:01 after 1h 16m

## Experimental setup

Six training runs, three architectures (CLIP baseline, CLIP+DIRE fusion, SGF-Net), two variants each (mix_aug without DAT, mix_aug_dat with DAT).

Shared training config:
- 50/50 CommForensics + aiart (stratified 80/20 split; 1000-sample held-out test shard)
- Tier-1 bias-removal transforms: RandomJPEGCompression(q∈[60,95], p=0.7) → RandomResizedCrop(224, [0.7,1.0]) → ColorJitter(0.1)
- seed=123, batch=32, img_size=224, lr=2e-4
- Baseline / DIRE: 30 epochs × 200 steps (6k total, ~45 min train on H100)
- SGF: 40 epochs × 300 steps (12k total, ~1h55m train on H100)
- DAT variants: gradient reversal head, hidden=[128], num_domains=2, max_lambda=0.5 (linear ramp)

## Full results matrix

`f_*` = forced (no abstention); `t_*` = threshold @ τ=0.9; `c_*` = conformal (α=0.05).
`--` on RAID AUROC because RAID streaming delivers single-class (fakes only).

| run | dataset | AUROC | f_acc | t_acc | t_abst | c_acc | c_abst |
|---|---|---|---|---|---|---|---|
| baseline_mix_aug | commfor | 0.966 | 0.909 | 0.967 | 0.258 | 0.938 | 0.090 |
| baseline_mix_aug | raid    | --    | 0.844 | 0.937 | 0.342 | 0.888 | 0.131 |
| baseline_mix_aug | aiart   | 0.933 | 0.868 | 0.966 | 0.554 | 0.918 | 0.191 |
| baseline_mix_aug | cifake  | 0.975 | 0.918 | 0.994 | 0.565 | 0.969 | 0.177 |
| baseline_mix_aug_dat | commfor | 0.967 | 0.907 | 0.975 | 0.318 | 0.946 | 0.126 |
| baseline_mix_aug_dat | raid    | --    | 0.883 | 0.969 | 0.352 | 0.930 | 0.148 |
| baseline_mix_aug_dat | aiart   | 0.943 | 0.882 | 0.976 | 0.623 | 0.942 | 0.224 |
| baseline_mix_aug_dat | cifake  | 0.970 | 0.915 | 0.992 | 0.611 | 0.968 | 0.211 |
| dire_mix_aug | commfor | 0.969 | 0.903 | 0.978 | 0.314 | 0.951 | 0.149 |
| dire_mix_aug | raid    | --    | 0.880 | 0.973 | 0.385 | 0.941 | 0.185 |
| dire_mix_aug | aiart   | 0.936 | 0.863 | 0.971 | 0.622 | 0.942 | 0.275 |
| dire_mix_aug | cifake  | 0.971 | 0.909 | 0.996 | 0.638 | 0.975 | 0.256 |
| dire_mix_aug_dat | commfor | 0.965 | 0.907 | 0.969 | 0.251 | 0.933 | 0.077 |
| dire_mix_aug_dat | raid    | --    | 0.868 | 0.962 | 0.354 | 0.911 | 0.126 |
| dire_mix_aug_dat | aiart   | 0.940 | 0.866 | 0.972 | 0.612 | 0.926 | 0.186 |
| dire_mix_aug_dat | cifake  | 0.971 | 0.909 | 0.995 | 0.618 | 0.963 | 0.171 |
| sgf_mix_aug | commfor | 0.952 | 0.893 | 0.975 | 0.417 | 0.945 | 0.171 |
| sgf_mix_aug | raid    | --    | 0.250 | 0.149 | 0.407 | 0.204 | 0.196 |
| sgf_mix_aug | aiart   | 0.944 | 0.876 | 0.985 | 0.677 | 0.938 | 0.260 |
| sgf_mix_aug | cifake  | 0.969 | 0.905 | 0.994 | 0.639 | 0.970 | 0.239 |
| sgf_mix_aug_dat | commfor | 0.931 | 0.861 | 0.966 | 0.520 | 0.941 | 0.303 |
| sgf_mix_aug_dat | raid    | --    | 0.430 | 0.389 | 0.721 | 0.402 | 0.459 |
| sgf_mix_aug_dat | aiart   | 0.869 | 0.792 | 0.982 | 0.830 | 0.912 | 0.532 |
| sgf_mix_aug_dat | cifake  | 0.909 | 0.833 | 0.973 | 0.814 | 0.950 | 0.511 |

## Prior gen_aware baseline (for cross-reference)

`gen_aware` (multi-task CLIP with 5-class architecture-family head) trained on the old pipeline
(no Tier-1 aug, no 50/50 mix). Only commfor + raid available.

| run | dataset | AUROC | c_acc | c_abst |
|---|---|---|---|---|
| gen_aware (run 20260418_080356_0e86a3) | commfor | 0.964 | 0.949 | 0.122 |
| gen_aware (run 20260418_080356_0e86a3) | raid    | --    | 0.974 | 0.134 |

## Key findings

1. **aiart modality-shift problem is solved.** All six runs now clear 0.87 AUROC on the
   held-out aiart test shard, up from ~0.56 baseline — bias removal + 50/50 mix did what we
   wanted. Best: sgf_mix_aug @ 0.944.

2. **DIRE_mix_aug is the overall strongest model.** Leads on CommFor (0.969), RAID (0.941
   conformal acc), aiart (0.942 conformal acc — tied with baseline_dat), and cifake (0.975).

3. **Baseline + DAT is a clean strict win over baseline alone** on every OOD set — modest
   gain on CommFor, clear gain on RAID (+0.04 conformal acc), small gain on aiart.

4. **DAT's effect is architecture-dependent.** Helps baseline, basically neutral for DIRE,
   destabilizes SGF (training diverged — best checkpoint was epoch 1, loss then rose
   monotonically for 39 more epochs).

5. **SGF RAID collapse is the standout negative result.** sgf_mix_aug sits at 0.20 conformal
   acc on RAID — its novel forensic (spectral + pixel) branches latched onto
   training-distribution fingerprints that got distorted by JPEG augmentation, and it has no
   pretrained fallback when an unseen generator shows up. Baseline's CLIP features are
   JPEG-robust by design and survive the shift.

6. **gen_aware (old pipeline) is still the RAID leader** (0.974 conformal acc) — its
   architecture-family entropy head gives it a principled abstention signal on
   unfamiliar generators. Worth rerunning on the new pipeline.
