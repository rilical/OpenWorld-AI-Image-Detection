#!/bin/bash
# CLIP+DIRE fusion + Tier-1 bias-removal + 50/50 CommFor+aiart mix (no DAT)

cd /ocean/projects/cis250202p/gyar/personal/dl/OpenWorld-AI-Image-Detection
source .venv/bin/activate

TAG="dire_mix_aug"
LOG="outputs/${TAG}.log"
mkdir -p outputs

echo "=== ${TAG} Training ==="
echo "Start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

python scripts/train_with_dire.py \
    --config configs/${TAG}.yaml \
    --device cuda \
    2>&1 | tee "${LOG}"

RUN=$(tail -1 "${LOG}")
echo "RUN=${RUN}"

echo ""; echo "=== Calibration ==="
python scripts/calibrate_temperature.py \
    --config configs/calibration.yaml \
    --ckpt "${RUN}/checkpoints/best.pt" \
    --run "${RUN}" \
    --device cuda 2>&1 | tee -a "${LOG}"

echo ""; echo "=== Conformal ==="
python scripts/build_conformal.py \
    --config configs/calibration.yaml \
    --ckpt "${RUN}/checkpoints/best.pt" \
    --temperature "${RUN}/calibration/temperature.json" \
    --run "${RUN}" \
    --device cuda 2>&1 | tee -a "${LOG}"

echo ""; echo "=== Eval CommunityForensics ==="
python scripts/evaluate.py --dataset commfor --config configs/eval_commfor.yaml \
    --run "${RUN}" --device cuda 2>&1 | tee -a "${LOG}"

echo ""; echo "=== Eval RAID ==="
python scripts/evaluate.py --dataset raid --config configs/eval_raid.yaml \
    --run "${RUN}" --device cuda 2>&1 | tee -a "${LOG}"

echo ""; echo "=== Eval aiart (held-out test) ==="
python scripts/evaluate.py --dataset aiart --config configs/eval_aiart_test.yaml \
    --run "${RUN}" --device cuda 2>&1 | tee -a "${LOG}"

echo ""; echo "=== Eval CIFAKE ==="
python scripts/evaluate.py --dataset cifake --config configs/eval_cifake.yaml \
    --run "${RUN}" --device cuda 2>&1 | tee -a "${LOG}"

echo ""; echo "=== Done ==="
echo "End: $(date)"
echo "Run: ${RUN}"
