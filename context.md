# context.md — Novel Model Design for Open-World AI Image Detection

## 1. Problem Recap and Grader Feedback

Our milestone scored 39/40. The one lost point was **novelty**: "What are you doing that others have not done yet?" Additional feedback flagged:

- Abstention overuse concern — need principled regulation, not just post-hoc thresholding
- Misleading figure (4-class output shown, only 2 exist)
- Need precision/recall/F1/confusion matrix, not just accuracy
- Need quantitative comparison with prior work

The current baseline (frozen CLIP ViT-B/32 + MLP) is essentially UnivFD (Ojha et al., CVPR 2023) with a slightly larger head. This is not novel. We need a genuinely new architecture.

---

## 2. Literature Landscape (what exists, what doesn't)

### Semantic-level detectors (CLIP-based)
| Method | Approach | Limitation |
|--------|----------|-----------|
| UnivFD (Ojha, CVPR 2023) | Frozen CLIP + linear probe | Misses low-level artifacts; degrades on SD3/DALL-E 3 |
| Community Forensics (Park, CVPR 2025) | CLIP + massive data scaling (4803 generators) | Data-scaling only, no architectural novelty |
| AIDE (Yan, ICLR 2025) | DCT+SRM filters ∥ CLIP, channel-concatenation fusion | Fixed fusion; handcrafted SRM filters |
| DeeCLIP (Keita, 2025) | CLIP-ViT + LoRA + triplet loss | No frequency branch; trained on ProGAN-era data |
| LNCLIP-DF (2025) | Fine-tune only LayerNorm (0.03%) + slerp augmentation | Pure metric learning; no forensic signal exploitation |

### Frequency / spectral detectors
| Method | Approach | Limitation |
|--------|----------|-----------|
| SPAI (Karageorgiou, CVPR 2025) | Masked spectral pretext + spectral context attention | Spectral only, no semantic branch |
| DEFEND (2025) | Weighted Fourier filter per frequency band | No multi-branch integration |
| Synthbuster | Cross-difference filter + FFT on residual | Training-free, limited adaptability |

### Pixel-level / structural detectors
| Method | Approach | Limitation |
|--------|----------|-----------|
| NPR (Tan, CVPR 2024) | Neighboring pixel differences expose upsampling artifacts | +12.8% improvement alone but never combined with CLIP or spectral features |
| DIRE (Wang, ICCV 2023) | Diffusion reconstruction error | Requires full diffusion pass at inference — prohibitively expensive |

### Abstention / selective prediction
| Method | Approach | Limitation |
|--------|----------|-----------|
| Selective Classification (Geifman, 2017) | Risk-coverage framework | No integration with forensic features |
| SCRC (Xu, 2025) | Conformal prediction + selective classification | General framework, never applied to forensics |
| Our milestone baseline | Post-hoc threshold on max-softmax | Abstention not learned; model not trained to know what it doesn't know |

### Test-time adaptation for forensics
| Method | Approach | Limitation |
|--------|----------|-----------|
| TTP-AP (2025) | Prototype projection to known domains at test time | Not combined with conformal guarantees |
| TENT | Entropy minimization on test batches | Generic, not forensic-specific |

---

## 3. Identified Gaps (our opportunity)

1. **No method jointly optimizes spectral, semantic, AND pixel-structural signals with learned adaptive fusion.** AIDE concatenates fixed DCT/SRM with CLIP. Nobody gates the fusion dynamically per-image.

2. **NPR features (+12.8% improvement, CVPR 2024) have never been combined with CLIP embeddings.** This is a free lunch waiting to be taken.

3. **No forensic detector trains with abstention awareness.** Every method trains binary CE then bolts on thresholding. Training the model to produce calibrated uncertainty from the start would make abstention far more effective.

4. **No frequency-aware attention mechanism dynamically weights which forensic branch to trust per input.** Different generators leave different artifacts (diffusion → spectral anomalies, GANs → pixel-level upsampling artifacts). A fixed fusion strategy cannot adapt.

5. **Spectral features are inherently more robust to adversarial perturbations** (which concentrate in specific frequency bands), but this robustness advantage is unexploited in current multi-branch designs.

---

## 4. Novel Model: Spectral-Gated Forensic Fusion Network (SGF-Net)

### 4.1 Core Idea

Three parallel branches extract complementary forensic signals at different abstraction levels. A **spectral gating network** learns to dynamically weight each branch per-image based on the spectral and structural characteristics of the input. The model is trained with a **confidence-calibrated auxiliary loss** that encourages well-separated confident-correct vs. uncertain-wrong predictions, making downstream abstention more effective.

### 4.2 Architecture

```
Input Image x ∈ R^(B×3×224×224)
    │
    ├─── Branch 1: Semantic (CLIP)
    │      CLIP ViT-B/32 encoder (frozen) → z_clip ∈ R^(B×512)
    │
    ├─── Branch 2: Spectral Forensics
    │      2D FFT → log-magnitude spectrum → 3-layer CNN → z_spec ∈ R^(B×128)
    │
    └─── Branch 3: Pixel Forensics (NPR + SRM)
           NPR: horizontal/vertical/diagonal pixel diffs (4ch)
           SRM: 3 predefined high-pass filters (3ch)
           Stack → 7-channel tensor → 3-layer CNN → z_pixel ∈ R^(B×128)
              │
              ▼
    ┌─────────────────────────────────────────────┐
    │  Spectral Gating Network                    │
    │                                             │
    │  stats = [spectral_stats, pixel_stats]      │
    │  g = softmax(MLP(stats)) ∈ R^(B×3)         │
    │                                             │
    │  z_fused = g₀·W_clip·z_clip                │
    │          + g₁·W_spec·z_spec                 │
    │          + g₂·W_pixel·z_pixel               │
    │                                             │
    │  z_fused ∈ R^(B×256)                        │
    └─────────────────────────────────────────────┘
              │
              ▼
    Classification Head: Linear(256,128) → ReLU → Dropout → Linear(128,2)
              │
              ▼
    Output: logits ∈ R^(B×2), probs = softmax(logits)
```

### 4.3 Branch Details

**Branch 1 — Semantic (existing, frozen CLIP)**
- Exactly the current `CLIPBinaryDetector.encode()` → 512-dim
- Captures high-level semantic tells (text artifacts, layout anomalies, CLIP-learned generation signatures)
- Frozen: no gradient flow through CLIP backbone

**Branch 2 — Spectral Forensics (new)**
- Compute `torch.fft.fft2` on each RGB channel → complex spectrum
- Take `log(1 + |spectrum|)` → 3-channel log-magnitude image (224×224×3)
- Shift zero-frequency to center (`torch.fft.fftshift`)
- 3-layer CNN:
  - Conv2d(3, 32, 3, padding=1) → BatchNorm → ReLU
  - Conv2d(32, 64, 3, stride=2, padding=1) → BatchNorm → ReLU
  - Conv2d(64, 128, 3, stride=2, padding=1) → BatchNorm → ReLU
  - AdaptiveAvgPool2d(1) → Flatten → 128-dim
- **Why this works**: Real images follow 1/f spectral falloff. Diffusion models produce anomalous spectral peaks. GANs create periodic grid artifacts in frequency domain. The CNN learns to detect these deviations.

**Branch 3 — Pixel Forensics (new, NPR-inspired)**
- NPR features (4 channels):
  - Horizontal: `|x[:,:,:-1] - x[:,:,1:]|` (padded to 224)
  - Vertical: `|x[:,:-1,:] - x[:,1:,:]|` (padded to 224)
  - Diagonal-right: `|x[:,:-1,:-1] - x[:,1:,1:]|` (padded)
  - Diagonal-left: `|x[:,:-1,1:] - x[:,1:,:-1]|` (padded)
- SRM high-pass filters (3 channels):
  - First-order edge: `[0,-1,0; -1,4,-1; 0,-1,0]`
  - Second-order: `[-1,2,-1; 2,-4,2; -1,2,-1]`
  - Third-order (SQUARE 5×5): standard SRM filter
- Stack → 7-channel input, same 3-layer CNN architecture as spectral branch → 128-dim
- **Why this works**: Generative models introduce correlated pixel patterns during upsampling (NPR, CVPR 2024 showed +12.8% improvement). SRM filters extract noise residuals that reveal processing pipeline fingerprints.

**Spectral Gating Network (the novel fusion mechanism)**
- Extract lightweight statistics from branch outputs:
  - Spectral stats: mean, std, kurtosis of frequency magnitudes across low/mid/high bands (9 values)
  - Pixel stats: mean, std of NPR activations per direction (8 values)
  - Total: 17-dim statistics vector
- Small MLP: Linear(17, 32) → ReLU → Linear(32, 3) → Softmax
- Produces per-sample gate weights g ∈ R^(B×3) that sum to 1
- Each branch embedding is projected to common dimension (256) before gated fusion:
  - `W_clip`: Linear(512, 256)
  - `W_spec`: Linear(128, 256)
  - `W_pixel`: Linear(128, 256)
  - `z_fused = g₀ * W_clip(z_clip) + g₁ * W_spec(z_spec) + g₂ * W_pixel(z_pixel)`
- **Why this is novel**: No existing method uses input-dependent gating to fuse forensic branches. AIDE uses fixed concatenation. DeeCLIP uses fixed fusion. The gate learns that diffusion-generated images are best detected via spectral anomalies while GAN-generated images are best detected via pixel artifacts, and routes accordingly.

### 4.4 Training Strategy

**Primary loss**: CrossEntropyLoss (compatible with existing training loop)

**Auxiliary loss — Confidence Separation Loss (new)**:
```
L_conf = -mean[ correct * log(confidence) + (1-correct) * log(1-confidence) ]
```
This is BCE applied to (confidence, correctness). It encourages the model to output high confidence when correct and low confidence when wrong. Unlike standard CE which only penalizes wrong class assignment, this explicitly shapes the confidence landscape for better downstream abstention.

**Combined loss**:
```
L_total = L_CE + λ * L_conf     (λ = 0.3, tunable)
```

**Training details**:
- Freeze CLIP encoder (Branch 1) — no gradient
- Train Branches 2 and 3 from scratch
- Train gating network from scratch
- Train classification head from scratch
- AdamW, lr=2e-4 (same as baseline), weight_decay=1e-4
- Optional: warm up branches independently for 2 epochs before enabling gating

### 4.5 Why This Is Novel (addressing grader concern directly)

| Contribution | Prior art | What we add |
|---|---|---|
| Multi-branch forensic fusion | AIDE (ICLR 2025) concatenates DCT+SRM∥CLIP | We add NPR pixel forensics as a third branch and use *learned adaptive gating* instead of fixed concatenation |
| Spectral-guided gating | No prior work | First use of spectral/structural statistics to dynamically weight forensic branch contributions per-image |
| NPR + CLIP integration | NPR (CVPR 2024) standalone; CLIP detectors standalone | First combination of neighboring-pixel-relationship features with CLIP semantic embeddings |
| Confidence-calibrated training | All prior detectors train binary CE only | Auxiliary loss shapes confidence surface to improve post-hoc calibration and abstention effectiveness |
| End-to-end with conformal abstention | Conformal prediction applied post-hoc everywhere | We train confidence-aware, then calibrate temperature, then apply conformal prediction — first integrated pipeline for forensic detection |

### 4.6 Compatibility with Existing Infrastructure

The model slots into the existing pipeline with minimal changes:
- **forward()** returns `{"logits": (B,2), "probs": softmax(logits)}` — same interface
- **Training loop**: needs small modification to support auxiliary loss (add `loss_fn` parameter to `run_training` or subclass)
- **Evaluation**: fully compatible — logits → temperature scaling → conformal prediction → abstention works identically
- **Inference/Predictor**: no changes needed
- **Config**: new YAML config `configs/sgf_net.yaml`
- **Dependencies**: no new packages — uses only `torch.fft`, `torch.nn`, existing deps

### 4.7 Expected Impact

Based on literature results:
- NPR alone gives +12.8% cross-generator improvement (Tan, CVPR 2024)
- Spectral features give +5.5% AUC improvement (SPAI, CVPR 2025)
- Learned fusion typically outperforms concatenation by 2-5% (various multi-branch works)
- Confidence-calibrated training should improve ECE and reduce AURC, making abstention more selective

Conservative estimate: **+8-15% selective accuracy improvement** on open-world benchmarks (VCT2, RAID) compared to the CLIP-only baseline, with more informative abstention (lower abstention rate at same accuracy, or higher accuracy at same coverage).

---

## 5. Ablation Plan

To demonstrate that each component contributes, we will evaluate:

| Ablation | Description |
|----------|-------------|
| A: CLIP-only (current baseline) | Frozen CLIP + MLP, no fusion |
| B: CLIP + Spectral (concat) | Two branches, naive concatenation |
| C: CLIP + Pixel (concat) | Two branches, naive concatenation |
| D: CLIP + Spectral + Pixel (concat) | Three branches, naive concatenation |
| E: SGF-Net (full, gated) | Three branches + spectral gating |
| F: SGF-Net + ConfLoss | Full model + confidence-calibrated training |

Metrics per ablation: AUROC, TPR@1%FPR, ECE, AURC, selective accuracy at 90% coverage, abstention rate at 95% selective accuracy.

---

## 6. Comparison with Prior Work

We will report numbers from:
- UnivFD (Ojha, CVPR 2023) — nearest architectural ancestor
- Community Forensics (Park, CVPR 2025) — SOTA data-scaling approach
- AIDE (Yan, ICLR 2025) — nearest multi-branch competitor
- NPR (Tan, CVPR 2024) — pixel forensics baseline
- VCT2 benchmark tables — detector accuracy on modern generators

This addresses the grader's request for "comparison with prior work" and "how your approach performs relative to existing methods."

---

## 7. Implementation Order

1. Implement `SpectralBranch` module
2. Implement `PixelForensicBranch` module (NPR + SRM)
3. Implement `SpectralGatingNetwork`
4. Implement `SGFNet` combining all components
5. Implement confidence separation loss
6. Modify training loop to support auxiliary loss
7. Create config `configs/sgf_net.yaml`
8. Create training script `scripts/train_sgf.py`
9. Train and evaluate on CommunityForensics
10. Run ablations (A through F)
11. Evaluate on VCT2, RAID, ARIA
12. Generate comparison tables and figures for final report

---

## 8. Round-2 Training Jobs Submitted (2026-04-18)

Six training jobs submitted to GPU-shared partition: 2 variants x 3 architectures. All use
Tier-1 bias-removal transforms + 50/50 CommFor+aiart mix. DAT variants add domain-adversarial
training via gradient reversal.

| Job ID    | Tag                      | Arch     | DAT | Wall  | Config                               |
|-----------|--------------------------|----------|-----|-------|--------------------------------------|
| 39826909  | bl_mix_aug               | baseline | no  | 2h    | configs/baseline_mix_aug.yaml        |
| 39826910  | bl_mix_aug_dat           | baseline | yes | 2h    | configs/baseline_mix_aug_dat.yaml    |
| 39826916  | dire_mix_aug             | dire     | no  | 2h    | configs/dire_mix_aug.yaml            |
| 39826917  | dire_mix_aug_dat         | dire     | yes | 2h    | configs/dire_mix_aug_dat.yaml        |
| 39826918  | sgf_mix_aug              | sgf      | no  | 4h    | configs/sgf_mix_aug.yaml             |
| 39826919  | sgf_mix_aug_dat          | sgf      | yes | 4h    | configs/sgf_mix_aug_dat.yaml         |

Train params: baseline/dire = 30 epochs x 200 steps (6k total); sgf = 40 epochs x 300 steps (12k).
Shared: batch=32, img=224, lr=2e-4, seed=123, max_train_samples=5000, calibration_fraction=0.1.

Post-train pipeline (per job): calibrate_temperature -> build_conformal ->
evaluate on commfor, raid, aiart-test (held-out 1000-sample split), cifake.

See reports/round2_mix_dat_plan.md for full motivation (shortcut learning, two OOD failure modes,
why Tier-1 / mix / DAT each address specific failure classes).
