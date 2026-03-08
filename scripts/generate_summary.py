"""Generate a markdown results summary from evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from _bootstrap import bootstrap_repo_source

bootstrap_repo_source()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate results summary from eval metrics")
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _fmt(value, digits: int = 4):
    if isinstance(value, dict):
        if "mean" in value:
            return f"{value['mean']:.{digits}f} [{value.get('lower', 0):.{digits}f}, {value.get('upper', 0):.{digits}f}]"
        return json.dumps(value, sort_keys=True)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if value is None:
        return "null"
    return str(value)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)
    eval_dir = run_dir / "eval"

    lines = [
        "# Results Summary",
        "",
        f"**Run**: `{run_dir.name}`",
        f"**Generated**: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]

    datasets = {}
    if eval_dir.exists():
        for dataset_dir in sorted(path for path in eval_dir.iterdir() if path.is_dir()):
            metrics_path = dataset_dir / "metrics.json"
            if metrics_path.exists():
                with metrics_path.open("r", encoding="utf-8") as f:
                    datasets[dataset_dir.name] = json.load(f)

    if not datasets:
        lines.append("No evaluation results found.")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(lines), encoding="utf-8")
        return

    lines.extend(
        [
            "## Core Metrics",
            "",
            "| Dataset | Mode | AUROC | TPR@1%FPR | ECE | Coverage | Abstain Rate | Selective Accuracy | AURC |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for dataset, modes in datasets.items():
        for mode, metrics in modes.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        dataset,
                        mode,
                        _fmt(metrics.get("auroc")),
                        _fmt(metrics.get("tpr_at_1pct_fpr")),
                        _fmt(metrics.get("ece")),
                        _fmt(metrics.get("coverage")),
                        _fmt(metrics.get("abstain_rate")),
                        _fmt(metrics.get("selective_accuracy")),
                        _fmt(metrics.get("aurc")),
                    ]
                )
                + " |"
            )
    lines.append("")

    lines.extend(
        [
            "## Bootstrap Confidence Intervals",
            "",
            "| Dataset | Mode | AUROC (95% CI) | Selective Accuracy (95% CI) |",
            "|---|---|---|---|",
        ]
    )
    for dataset, modes in datasets.items():
        for mode, metrics in modes.items():
            lines.append(
                f"| {dataset} | {mode} | {_fmt(metrics.get('bootstrap_auroc'))} | {_fmt(metrics.get('bootstrap_selective_accuracy'))} |"
            )
    lines.append("")

    lines.extend(
        [
            "## Empirical Conformal Coverage",
            "",
            "| Dataset | Mode | Overall | Class 0 | Class 1 |",
            "|---|---|---|---|---|",
        ]
    )
    for dataset, modes in datasets.items():
        for mode, metrics in modes.items():
            coverage = metrics.get("conformal_coverage", {})
            conditional = coverage.get("class_conditional", {})
            lines.append(
                f"| {dataset} | {mode} | {_fmt(coverage.get('overall'))} | {_fmt(conditional.get('0'))} | {_fmt(conditional.get('1'))} |"
            )
    lines.append("")

    for dataset, modes in datasets.items():
        for mode, metrics in modes.items():
            if metrics.get("worst_group_selective_accuracy"):
                worst = metrics["worst_group_selective_accuracy"]
                lines.append(
                    f"**{dataset}/{mode} worst-group selective accuracy**: `{worst.get('worst_group', '?')}` -> {_fmt(worst.get('worst'))}"
                )
    lines.append("")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary written to {args.out}")


if __name__ == "__main__":
    main()
