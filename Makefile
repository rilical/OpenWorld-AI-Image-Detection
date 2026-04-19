.PHONY: smoke test lint-placeholder train-baseline train-dire calibrate conformal eval-commfor eval-vct2 eval-raid eval-aria plots cache-residuals predict demo summary

RUN     ?= outputs/runs/latest
CONFIG  ?= configs/baseline_clip.yaml
DEVICE  ?= cpu

smoke:
	python3 -m compileall src

test:
	python3 -m pytest -q

lint-placeholder:
	@echo "No linter configured yet."

train-baseline:
	python3 scripts/train_baseline.py --config $(CONFIG) --device $(DEVICE)

train-dire:
	python3 scripts/train_with_dire.py --config configs/dire_fusion.yaml --device $(DEVICE)

calibrate:
	python3 scripts/calibrate_temperature.py --config $(CONFIG) \
		--ckpt $(RUN)/checkpoints/best.pt --run $(RUN) --device $(DEVICE)

conformal:
	python3 scripts/build_conformal.py --config $(CONFIG) \
		--ckpt $(RUN)/checkpoints/best.pt \
		--temperature $(RUN)/calibration/temperature.json \
		--run $(RUN) --device $(DEVICE)

eval-commfor:
	python3 scripts/evaluate.py --dataset commfor --config configs/eval_commfor.yaml --run $(RUN) --device $(DEVICE)

eval-vct2:
	python3 scripts/evaluate.py --dataset vct2 --config configs/eval_vct2.yaml --run $(RUN) --device $(DEVICE)

eval-raid:
	python3 scripts/evaluate.py --dataset raid --config configs/eval_raid.yaml --run $(RUN) --device $(DEVICE)

eval-aria:
	python3 scripts/evaluate.py --dataset aria --config configs/eval_aria.yaml --run $(RUN) --device $(DEVICE)

plots:
	python3 scripts/make_plots.py --runs outputs/runs --out reports/figures

cache-residuals:
	python3 scripts/cache_residuals.py --config configs/dire_fusion.yaml --run $(RUN) --device $(DEVICE)

predict:
	python3 scripts/predict_image.py --run $(RUN) --image $(IMAGE) --device $(DEVICE)

demo:
	python3 demo/app.py --run $(RUN) --device $(DEVICE)

summary:
	python3 scripts/generate_summary.py --run $(RUN) --out reports/results_summary.md
