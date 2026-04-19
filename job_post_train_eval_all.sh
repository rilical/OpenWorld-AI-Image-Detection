#!/bin/bash
# Run calibrate -> conformal -> 4 evals for all 6 mix_aug* runs.

cd /ocean/projects/cis250202p/gyar/personal/dl/OpenWorld-AI-Image-Detection
source .venv/bin/activate

mkdir -p outputs
LOG="outputs/post_train_eval_all.log"
: > "${LOG}"

RUNS=(
    "outputs/runs/baseline_mix_aug"
    "outputs/runs/baseline_mix_aug_dat"
    "outputs/runs/dire_mix_aug"
    "outputs/runs/dire_mix_aug_dat"
    "outputs/runs/sgf_mix_aug"
    "outputs/runs/sgf_mix_aug_dat"
)

echo "=== Post-train pipeline over ${#RUNS[@]} runs ===" | tee -a "${LOG}"
echo "Start: $(date)" | tee -a "${LOG}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')" | tee -a "${LOG}"

for RUN in "${RUNS[@]}"; do
    TAG=$(basename "${RUN}")
    echo "" | tee -a "${LOG}"
    echo "############################################" | tee -a "${LOG}"
    echo "# ${TAG}  (RUN=${RUN})" | tee -a "${LOG}"
    echo "############################################" | tee -a "${LOG}"

    if [ ! -f "${RUN}/checkpoints/best.pt" ]; then
        echo "[skip] no best.pt found at ${RUN}/checkpoints/best.pt" | tee -a "${LOG}"
        continue
    fi

    echo ">> Calibration" | tee -a "${LOG}"
    python scripts/calibrate_temperature.py \
        --config configs/calibration.yaml \
        --ckpt "${RUN}/checkpoints/best.pt" \
        --run "${RUN}" \
        --device cuda 2>&1 | tee -a "${LOG}"

    echo ">> Conformal" | tee -a "${LOG}"
    python scripts/build_conformal.py \
        --config configs/calibration.yaml \
        --ckpt "${RUN}/checkpoints/best.pt" \
        --temperature "${RUN}/calibration/temperature.json" \
        --run "${RUN}" \
        --device cuda 2>&1 | tee -a "${LOG}"

    echo ">> Eval CommunityForensics" | tee -a "${LOG}"
    python scripts/evaluate.py --dataset commfor --config configs/eval_commfor.yaml \
        --run "${RUN}" --device cuda 2>&1 | tee -a "${LOG}"

    echo ">> Eval RAID" | tee -a "${LOG}"
    python scripts/evaluate.py --dataset raid --config configs/eval_raid.yaml \
        --run "${RUN}" --device cuda 2>&1 | tee -a "${LOG}"

    echo ">> Eval aiart (held-out test)" | tee -a "${LOG}"
    python scripts/evaluate.py --dataset aiart --config configs/eval_aiart_test.yaml \
        --run "${RUN}" --device cuda 2>&1 | tee -a "${LOG}"

    echo ">> Eval CIFAKE" | tee -a "${LOG}"
    python scripts/evaluate.py --dataset cifake --config configs/eval_cifake.yaml \
        --run "${RUN}" --device cuda 2>&1 | tee -a "${LOG}"
done

echo "" | tee -a "${LOG}"
echo "=== Done ===" | tee -a "${LOG}"
echo "End: $(date)" | tee -a "${LOG}"
