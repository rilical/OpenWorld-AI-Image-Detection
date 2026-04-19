# Round-2 Experimental Plan — Bias Removal + Mixed Training + DAT

_CMU 18-786, Open-World AI Image Detection — 2026-04-18_

This document motivates and specifies the six training runs launched on 2026-04-18 that extend the three existing detectors (CLIP baseline, CLIP+DIRE fusion, SGF-Net) with three coordinated interventions: Tier-1 bias-removal data augmentation, 50/50 mixed-domain training, and domain-adversarial training (DAT). It is intended both as engineering notes and as the source for the "Methods" and "OOD Analysis" sections of the final report. The original SGF-Net design doc remains in `context.md`.

---

## 1. Why these interventions are needed — the problem with our current results

### 1.1 Headline numbers (pre-mix, CommFor-only training)

All three models were trained on CommunityForensics-Small (CommFor), calibrated with temperature scaling + split conformal prediction (α=0.05), and evaluated on three benchmarks. Evaluation is "forced" (no abstention) unless noted.

| Model              | CommFor acc | CommFor AUROC | CommFor TPR@1%FPR | RAID TPR | aiart acc | aiart AUROC |
| ------------------ | ----------- | ------------- | ----------------- | -------- | --------- | ----------- |
| Baseline CLIP      | **96.96%**  | 0.9957        | 94.7%             | **98.04%** | 56.32%    | 0.5648      |
| CLIP + DIRE fusion | 96.94%      | 0.9952        | 93.9%             | **98.50%** | 55.54%    | 0.5579      |
| SGF-Net            | 89.72%      | 0.9572        | 67.3%             | 81.82%   | 56.86%    | 0.6153      |

The 3-model unanimous-abstain ensemble further improves in-distribution: 98.61% CommFor accuracy at 91.4% coverage, 98.92% RAID TPR at 78.1% coverage.

Two facts jump out:

1. **CommFor / RAID are essentially saturated.** Baseline and DIRE both hit ~97% on CommFor and ~98% TPR on RAID. There is no headroom left on these benchmarks — any further work targeting them is measurement noise, not science.
2. **All three models collapse on aiart** (Hemg/AI-Generated-vs-Real-Images-Datasets — 152k stylized AI art vs real museum-style art; we use a seed=123 5000-sample subset). Accuracy sits in 55-57% — barely above random. This is the opposite of what we want from an "open-world" detector.

### 1.2 What the aiart collapse tells us — per-model breakdown

| Model              | aiart acc | aiart AUROC | What the number means                                                                                                                                    |
| ------------------ | --------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Baseline CLIP      | 56.32%    | 0.5648      | Near random. CLIP semantic features + MLP learned a decision boundary that does not transfer to paintings/renderings at all.                             |
| CLIP + DIRE fusion | 55.54%    | 0.5579      | **Worse than baseline.** The DIRE branch actively hurts: its reconstruction-residual signal was tuned for diffusion-generated photos and gives a misleading prior on non-diffusion / stylized images. |
| SGF-Net            | 56.86%    | 0.6153      | Best AUROC of the three but still far from usable. The spectral branch picks up a faint signal — paintings have very different FFT statistics from photos, so the frequency branch does some work — but the pixel-forensic branch was trained on photo artifacts and contributes noise. |

Importantly, all three models fail at roughly the same accuracy. That tells us the failure is **not architectural** — it's a training-data-distribution failure. No detector trained only on CommFor photos can tell whether a painting is AI-made, because the features learned are specific to the photorealistic regime.

### 1.3 Two distinct OOD failure modes

Our benchmarks probe two different kinds of "open-world" failure:

- **Unseen-generator OOD** (RAID): the detector sees a new generator architecture at test time, but the images are still photographs in the same modality as training. All three models pass this (98% TPR). This is what most detection papers call "generalization."
- **Modality-shift OOD** (aiart): the detector sees a fundamentally different image modality (paintings vs. photos). All three models fail. This is what our project explicitly targets — the "open-world" in our title — and the current numbers show we have not solved it.

Our final report needs to do more than show the unseen-generator numbers that competitors also show. It needs to directly measure and address the modality-shift failure. The three interventions below do exactly that.

---

## 2. Intervention #1 — Tier-1 bias-removal data augmentation

### 2.1 Why we suspect the CommFor numbers are inflated (shortcut learning)

CommunityForensics-Small stores real photos with their original JPEG compression history intact (they come from web sources), while AI-generated fakes are produced by sampling pipelines that usually output PNG or in-memory uncompressed tensors. A detector trained on this corpus is free to exploit the trivial correlation:

> **shortcut:** has-JPEG-artifacts → real · no-JPEG-artifacts → fake

This is a textbook case of **shortcut learning** (Geirhos et al., 2020 — "Shortcut Learning in Deep Neural Networks"). The model learns a feature that perfectly separates the training data but has no causal relationship with whether the image was AI-generated.

The aiart collapse is circumstantial evidence that this is happening. aiart images — both the real artwork and the AI art — are distributed as standard web JPEGs, so the JPEG-history shortcut provides no signal. If the detector had genuinely learned "is AI-generated," aiart should still be partially solvable (stylized AI art is semantically quite different from real paintings). The fact that it isn't suggests the detector is leaning heavily on a pipeline-specific bias that does not exist in aiart.

### 2.2 Related biases we also want to neutralize

JPEG history is the most-documented shortcut but not the only candidate:

- **Color-histogram bias**: synthesis pipelines often produce slightly different tonal distributions than camera sensors; a detector can learn "saturation > X" rather than the actual artifact signature.
- **Spatial-composition bias**: AI-generated images are often framed at center-crops (because they come out of square-aspect-ratio generators) while web photos have varied framing — resolution/crop priors leak class information.

### 2.3 The Tier-1 augmentation pipeline

Applied only during training (eval pipeline is unchanged so we preserve comparability with earlier numbers):

- **RandomJPEGCompression**: with p=0.7, re-encode the input as JPEG at quality ∈ [60, 95], then decode. This forces *both* classes to carry JPEG-like artifacts at training time, erasing the shortcut.
- **RandomResizedCrop(224, scale=[0.7, 1.0])**: disrupts the framing / center-crop bias. Replaces the deterministic resize-then-center-crop used in the baseline pipeline.
- **ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)**: disrupts low-level color-statistic bias. Deliberately small — we want to neutralize shortcuts, not destroy the signal.

We call this "Tier-1" because it's the lightest intervention: three cheap PIL-level transforms, no model changes. Tier-2 (not attempted in this round) would target frequency-domain biases with spectral masking or FFT-band dropouts.

### 2.4 What we expect

- CommFor accuracy should *drop* slightly when the JPEG shortcut disappears — this is a feature, not a bug. Whatever remains is the model's genuine signal.
- RAID performance should be roughly unchanged (RAID never had the JPEG shortcut to begin with — it uses real photos throughout).
- aiart should improve modestly if our shortcut hypothesis is correct.

---

## 3. Intervention #2 — 50/50 mixed-domain training (CommFor + aiart)

### 3.1 Why bias removal alone is not enough

Tier-1 augmentation removes *spurious* cues but it cannot teach the model what it hasn't seen. Every single training image for our current detectors is a photograph. No amount of JPEG augmentation will teach a CLIP detector that a Renaissance-style painting can be AI-generated — because in training, paintings simply don't exist. The features the detector learns are photo-specific whether or not there is a shortcut.

### 3.2 The trade-off we're explicitly making

There are two ways to get the model to learn painting-style features:

- **(A) Keep aiart OOD, try harder architectures** (self-supervised pretraining, contrastive representation learning, test-time adaptation). Compute-intensive, multi-week engineering.
- **(B) Mix aiart into training**. Cheap, fast, and directly exposes the model to the missing modality.

We take (B) because it fits within the project timeline. The trade-off: aiart is no longer an OOD benchmark after this round — we can no longer use it to measure generalization to unseen modalities. To compensate we bring in a *new* OOD dataset (CIFAKE — 120k CIFAR-10-style 32×32 images, 50/50 real/fake, with a different generator distribution and a different resolution). CIFAKE becomes the new modality-shift frontier.

### 3.3 Why 50/50 and not 80/20

- At 80/20 (CommFor-heavy), the mixed gradient is still dominated by CommFor photos; aiart acts more like noise than a second domain. We'd get most of the aiart deficit without closing it.
- At 50/50, each batch is equally weighted across domains; the feature extractor is forced to learn representations that work for both. This is the standard choice in multi-domain detection work.
- At pure 50/50 of the raw datasets we'd oversample aiart (it's much smaller after our 5k subsampling); we use a `WeightedRandomSampler` to keep *effective* exposure at 50/50 per batch.

### 3.4 Stratified 80/20 aiart split

We split the 5000-sample aiart subset into 4000 train / 1000 test, stratified by class, with seed=123 (matches the seeding used everywhere else). This lets us still report an aiart metric — now an in-distribution within-dataset metric — and catch overfitting to the specific 4000 training paintings.

---

## 4. Intervention #3 — Domain-Adversarial Training (DAT)

### 4.1 Why mixing alone is not enough

Mixing CommFor and aiart into training teaches the model both domains, but it doesn't *force* the learned representations to be shared. In practice, a sufficiently flexible model can just learn two separate decision boundaries — one for photos, one for paintings — inside the same network. At test time this model will still fail on CIFAKE (a third, unseen modality: 32×32 CIFAR-10-style images), because it has no mechanism that encourages features to generalize *beyond* the two domains it was trained on.

### 4.2 What DAT does (Ganin & Lempitsky, ICML 2015)

We attach a second head — a **domain classifier** — to the same feature extractor that produces task logits, connected via a **gradient reversal layer** (GRL). During the forward pass, GRL is the identity. During the backward pass, it multiplies gradients by −λ. Concretely:

- The domain classifier tries to predict which dataset each sample came from (CommFor=0, aiart=1).
- Its loss pushes the *domain head's* weights to get good at this classification.
- But because of the reversed sign on the gradients flowing back through GRL, its loss simultaneously pushes the *feature extractor's* weights to make domain classification *harder*.
- The equilibrium of this minimax game is a feature extractor that retains information useful for the task (AI-vs-real) while discarding information that identifies the domain.

In other words: DAT actively regularizes the learned representation to be **domain-invariant**.

### 4.3 Why this should help on CIFAKE specifically

If the feature space is invariant across {CommFor photos, aiart paintings}, it has been trained to suppress at least two very different modality signatures. That's no guarantee it will suppress a third unseen one (CIFAKE is 32×32 — a resolution gap much larger than photos-vs-paintings), but it's strictly more likely than a non-DAT model, which is free to keep any domain-specific feature that happens to help in training.

### 4.4 Training schedule — λ ramp from 0 to 0.5

- **Why ramp and not constant?** Early in training the task head has not yet learned the core AI-vs-real signal; if DAT kicks in at full strength immediately, the adversarial pressure will prevent the network from ever specializing enough to be useful. Standard practice (Ganin 2015, subsequent domain-adaptation literature) is to linearly ramp λ from 0 to its final value across training.
- **Why final λ = 0.5 and not 1.0?** λ=1.0 puts equal weight on domain-invariance and task accuracy. With only two source domains, that's too aggressive — the model can satisfy the domain-invariance constraint trivially by learning random features and still have a 50% chance on each domain. λ=0.5 keeps task accuracy as the dominant objective while adding a meaningful but non-destructive regularization term.
- **Only task loss uses the ramp**; domain loss is computed every step without ramp. GRL already scales the domain gradient by λ, so the domain loss coefficient in the scalar sum is 1.0 (no double-scaling).

### 4.5 Why we need domain labels on every training sample

DAT requires every sample to carry a domain ID. We extend both the `CommunityForensicsSmallDataset` and `AIArtDataset` return dicts with a `"domain_label"` field (0 and 1 respectively). At eval time the domain head output is ignored — we only consume `logits` — so held-out benchmarks (RAID, aiart-test, CIFAKE) can be evaluated without any change to the evaluation pipeline.

---

## 5. Why these three in combination, and not independently

Each intervention targets a different failure mode, and none of them is individually sufficient:

| Intervention              | Targets                          | Without it                                                                                   |
| ------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------- |
| Tier-1 augmentation       | Preprocessing/pipeline shortcuts | Model re-learns the JPEG shortcut in training even with mixed domains; apparent gains are spurious. |
| 50/50 mix training        | Missing-modality coverage        | Model is blind to paintings regardless of how clean the rest of training is.                 |
| DAT                       | Domain-specific representations  | Mixed-training model can learn two silos; will not generalize to a third OOD domain.         |

The factorial 2×3 design below separates their contributions: the marginal effect of DAT is measured by pairing each model's `*_mix_aug` run with its `*_mix_aug_dat` twin, holding Tier-1 + mix constant.

---

## 6. Experimental matrix — six new training runs

| Run                    | Architecture      | Tier-1 | 50/50 mix | DAT | Wall-clock |
| ---------------------- | ----------------- | ------ | --------- | --- | ---------- |
| baseline_mix_aug       | CLIP baseline     | ✓      | ✓         | —   | 2 h        |
| baseline_mix_aug_dat   | CLIP baseline     | ✓      | ✓         | ✓   | 2 h        |
| dire_mix_aug           | CLIP + DIRE       | ✓      | ✓         | —   | 2 h        |
| dire_mix_aug_dat       | CLIP + DIRE       | ✓      | ✓         | ✓   | 2 h        |
| sgf_mix_aug            | SGF-Net           | ✓      | ✓         | —   | 4 h        |
| sgf_mix_aug_dat        | SGF-Net           | ✓      | ✓         | ✓   | 4 h        |

SGF-Net gets double the wall-clock because in the previous round it underperformed baseline and DIRE on in-distribution metrics (89.72% CommFor vs. 96.94%). The leading diagnosis is undertraining: the three-branch fusion plus the learned spectral gating requires more optimization steps than a simple linear probe on CLIP. 4 h gives it roughly twice the total steps at the same batch size.

### Evaluation matrix per run (same four benchmarks for all six runs)

1. **CommFor-Eval** — in-distribution check. Numbers should stay close to the pre-mix baseline; a large drop indicates Tier-1 broke too much signal.
2. **RAID** — unseen-generator OOD. Should stay above 90% TPR; this validates that mixed training didn't hurt the generator-generalization story.
3. **aiart-test** (the held-out 20%) — in-distribution-within-mixed check. Success here is *necessary but not sufficient* — it shows the model can learn paintings when shown paintings.
4. **CIFAKE** — new modality-shift OOD. This is the critical test. If DAT works, `*_mix_aug_dat` should beat `*_mix_aug` by a clear margin here even when both match on the other three benchmarks.

### Decision criteria for the final report

- If Tier-1 costs us >2% on CommFor but unlocks >20% on aiart-test, the augmentation is worth it.
- If mix training delivers aiart-test ≈ CommFor parity, the modality-coverage hypothesis is confirmed.
- If DAT's CIFAKE gain > 3% over non-DAT while matching on other benchmarks, we have direct evidence that domain-invariant features transfer to unseen modalities.
- If DAT makes no CIFAKE difference, we report it as a negative result (mix training alone recovers most of the available signal).

---

## 7. Invariants

- seed=123 for every new run (matches all earlier runs; results are directly comparable).
- Temperature calibration + split conformal (α=0.05) applied post-hoc to each run as before.
- In-flight gen-aware job **39806889** must keep running — do not cancel or modify.
- The `save_checkpoint(...extra=…)` → `cfg=…` fix in `scripts/train_gen_aware.py` stays.
