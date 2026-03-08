"""Generate publication-quality plots from saved run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from _bootstrap import bootstrap_repo_source

bootstrap_repo_source()

import matplotlib.pyplot as plt
import numpy as np


KNOWN_MODES = ["forced", "temperature", "threshold", "conformal"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-quality plots from eval metrics")
    parser.add_argument("--runs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--style", choices=["default", "publication"], default="default")
    parser.add_argument("--format", default="png")
    parser.add_argument("--metrics", nargs="*", default=["auroc", "tpr_at_1pct_fpr", "ece", "aurc", "abstain_rate"])
    parser.add_argument("--tables-dir", default=None)
    return parser.parse_args()


def _set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def _save_fig(fig, out_dir: Path, name: str, formats: list[str]) -> None:
    for fmt in formats:
        fig.savefig(out_dir / f"{name}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def _save_table(rows: list[dict], tables_dir: Path, name: str) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    with (tables_dir / f"{name}.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    if rows:
        with (tables_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _normalize_metrics(payload: dict) -> dict[str, dict]:
    if any(key in payload for key in KNOWN_MODES):
        return {mode: payload[mode] for mode in KNOWN_MODES if mode in payload}
    return {"default": payload}


def _load_all_metrics(run_root: Path) -> dict:
    all_metrics = {}
    for run in sorted(path for path in run_root.iterdir() if path.is_dir()):
        run_payload = {}
        for metrics_path in sorted(run.glob("eval/*/metrics.json")):
            dataset = metrics_path.parent.name
            with metrics_path.open("r", encoding="utf-8") as f:
                run_payload[dataset] = _normalize_metrics(json.load(f))
        if run_payload:
            all_metrics[str(run)] = run_payload
    return all_metrics


def plot_risk_coverage(all_metrics: dict, out_dir: Path, formats: list[str], tables_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows = []
    for run_name, datasets in all_metrics.items():
        run_label = Path(run_name).name
        for dataset_name, mode_metrics in datasets.items():
            for mode, metrics in mode_metrics.items():
                rc = metrics.get("risk_coverage")
                if not rc:
                    continue
                ax.plot(rc["coverage"], rc["risk"], label=f"{run_label}/{dataset_name}/{mode}", linewidth=1.3)
                rows.append(
                    {
                        "run": run_label,
                        "dataset": dataset_name,
                        "mode": mode,
                        "aurc": metrics.get("aurc"),
                    }
                )
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk")
    ax.set_title("Risk-Coverage Curves")
    if rows:
        ax.legend(fontsize=7, loc="upper right")
    _save_fig(fig, out_dir, "risk_coverage", formats)
    _save_table(rows, tables_dir, "risk_coverage")


def plot_reliability(all_metrics: dict, out_dir: Path, formats: list[str], tables_dir: Path) -> None:
    for run_name, datasets in all_metrics.items():
        run_label = Path(run_name).name
        for dataset_name, mode_metrics in datasets.items():
            for mode, metrics in mode_metrics.items():
                reliability = metrics.get("reliability")
                bins = None if reliability is None else reliability.get("bins")
                if not bins:
                    continue
                acc = np.asarray(bins["accuracy"])
                edges = np.asarray(bins["bin_edges"])
                support = np.asarray(bins["support"])
                centers = (edges[:-1] + edges[1:]) / 2
                width = (edges[1] - edges[0]) * 0.8

                fig, ax = plt.subplots(figsize=(5, 4))
                ax.bar(centers, acc, width=width, alpha=0.7, color="#4C72B0")
                ax.plot([0, 1], [0, 1], "k--", linewidth=1)
                ax.set_xlabel("Confidence")
                ax.set_ylabel("Accuracy")
                ax.set_title(f"Reliability - {run_label}/{dataset_name}/{mode}")
                _save_fig(fig, out_dir, f"reliability_{run_label}_{dataset_name}_{mode}", formats)

                rows = [
                    {
                        "run": run_label,
                        "dataset": dataset_name,
                        "mode": mode,
                        "bin_center": float(center),
                        "accuracy": float(value),
                        "support": float(weight),
                    }
                    for center, value, weight in zip(centers, acc, support)
                ]
                _save_table(rows, tables_dir, f"reliability_{run_label}_{dataset_name}_{mode}")


def plot_summary_bars(all_metrics: dict, metric_names: list[str], out_dir: Path, formats: list[str], tables_dir: Path) -> None:
    datasets = sorted({dataset for run_payload in all_metrics.values() for dataset in run_payload})
    summary_rows = []

    for metric in metric_names:
        labels = []
        values_by_dataset = {dataset: [] for dataset in datasets}
        for run_name, run_payload in all_metrics.items():
            run_label = Path(run_name).name
            for mode in KNOWN_MODES:
                if not any(mode in ds_payload for ds_payload in run_payload.values()):
                    continue
                labels.append(f"{run_label}/{mode}")
                for dataset in datasets:
                    metric_payload = run_payload.get(dataset, {}).get(mode, {})
                    value = metric_payload.get(metric)
                    values_by_dataset[dataset].append(0.0 if value is None else value)
                    summary_rows.append(
                        {
                            "run": run_label,
                            "dataset": dataset,
                            "mode": mode,
                            "metric": metric,
                            "value": value,
                        }
                    )

        if not labels:
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4.5))
        x = np.arange(len(labels))
        width = 0.8 / max(len(datasets), 1)
        for idx, dataset in enumerate(datasets):
            ax.bar(x + idx * width, values_by_dataset[dataset], width=width, label=dataset)
        ax.set_xticks(x + width * max(len(datasets) - 1, 0) / 2)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by run/mode")
        ax.legend(fontsize=8)
        fig.tight_layout()
        _save_fig(fig, out_dir, f"summary_{metric}", formats)

    _save_table(summary_rows, tables_dir, "summary_metrics")


def plot_per_generator(all_metrics: dict, out_dir: Path, formats: list[str], tables_dir: Path) -> None:
    rows = []
    for run_name, datasets in all_metrics.items():
        run_label = Path(run_name).name
        if "vct2" not in datasets:
            continue
        generator_payload = datasets["vct2"].get("conformal") or next(iter(datasets["vct2"].values()), {})
        per_generator = generator_payload.get("per_generator")
        if not per_generator:
            continue

        ordered = sorted(
            ((name, payload.get("selective_accuracy", 0.0)) for name, payload in per_generator.items()),
            key=lambda item: item[1],
        )
        fig, ax = plt.subplots(figsize=(7, max(3, len(ordered) * 0.35)))
        ax.barh([name for name, _ in ordered], [value for _, value in ordered], color="#4C72B0")
        ax.set_xlabel("Selective Accuracy")
        ax.set_title(f"VCT2 per-generator - {run_label}")
        fig.tight_layout()
        _save_fig(fig, out_dir, f"vct2_per_generator_{run_label}", formats)

        for name, value in ordered:
            rows.append({"run": run_label, "generator": name, "selective_accuracy": value})

    _save_table(rows, tables_dir, "vct2_per_generator")


def main() -> None:
    args = parse_args()
    run_root = Path(args.runs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = Path(args.tables_dir) if args.tables_dir else Path(args.out).parent / "tables"
    formats = [fmt.strip() for fmt in args.format.split(",") if fmt.strip()]

    if args.style == "publication":
        _set_publication_style()

    all_metrics = _load_all_metrics(run_root)
    if not all_metrics:
        print(f"No metrics found in {run_root}")
        return

    plot_risk_coverage(all_metrics, out_dir, formats, tables_dir)
    plot_reliability(all_metrics, out_dir, formats, tables_dir)
    plot_summary_bars(all_metrics, args.metrics, out_dir, formats, tables_dir)
    plot_per_generator(all_metrics, out_dir, formats, tables_dir)
    print(f"Plots saved to {out_dir}")
    print(f"Tables saved to {tables_dir}")


if __name__ == "__main__":
    main()
