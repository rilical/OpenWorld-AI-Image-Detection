#!/bin/bash

cd /ocean/projects/cis250202p/gyar/personal/dl/OpenWorld-AI-Image-Detection

source .venv/bin/activate

python scripts/train_baseline.py --config configs/baseline_clip.yaml --device cuda > out.log 2>&1
