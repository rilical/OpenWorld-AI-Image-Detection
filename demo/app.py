"""Gradio demo for tri-state AI image detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _list_runs(runs_root: str) -> list[str]:
    root = Path(runs_root)
    if not root.exists():
        return []
    return sorted(str(path) for path in root.iterdir() if path.is_dir())


def build_demo(run_dir: str | None = None, runs_root: str = "outputs/runs", device: str = "cpu"):
    """Build a Gradio Blocks demo using the shared inference API."""
    import gradio as gr
    from owaid.inference import load_run

    available_runs = _list_runs(runs_root)
    selected_run = run_dir or (available_runs[0] if available_runs else "")
    predictor_cache = {}

    def get_predictor(selected: str):
        if not selected:
            raise ValueError("No run directory selected.")
        if selected not in predictor_cache:
            predictor_cache[selected] = load_run(selected, device)
        return predictor_cache[selected]

    def predict(selected: str, image):
        if image is None or not selected:
            return {}, 0.0, [], "", None
        predictor = get_predictor(selected)
        result = predictor.predict_pil(image)
        heatmap = predictor.render_residual_heatmap(image)
        return (
            {result["tri_state_label"]: result["confidence"]},
            result["confidence"],
            result["prediction_set"],
            "Yes" if result["abstained"] else "No",
            heatmap,
        )

    with gr.Blocks(title="Open-World AI Image Detection") as demo:
        gr.Markdown("# Open-World AI Image Detection with Abstention")
        gr.Markdown("Choose a saved run, upload an image, and inspect the tri-state decision.")

        run_selector = gr.Dropdown(
            choices=available_runs,
            value=selected_run or None,
            label="Run Directory",
            allow_custom_value=True,
        )
        image_input = gr.Image(type="pil", label="Upload Image")

        with gr.Row():
            label_output = gr.Label(label="Prediction")
            confidence_output = gr.Number(label="Calibrated Confidence", precision=4)
            set_output = gr.JSON(label="Prediction Set")
            abstained_output = gr.Textbox(label="Abstained", interactive=False)

        heatmap_output = gr.Image(label="Residual Heatmap (if available)")

        image_input.change(
            predict,
            inputs=[run_selector, image_input],
            outputs=[label_output, confidence_output, set_output, abstained_output, heatmap_output],
        )
        run_selector.change(
            predict,
            inputs=[run_selector, image_input],
            outputs=[label_output, confidence_output, set_output, abstained_output, heatmap_output],
        )

    return demo


def run_cli(image_path: str, run_dir: str, device: str = "cpu"):
    """CLI fallback: single image inference without Gradio."""
    from PIL import Image
    from owaid.inference import load_run

    predictor = load_run(run_dir, device)
    image = Image.open(image_path).convert("RGB")
    result = predictor.predict_pil(image)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Open-World AI Image Detection Demo")
    parser.add_argument("--run", default=None, help="Optional default run directory")
    parser.add_argument("--runs-root", default="outputs/runs", help="Directory containing saved runs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image", default=None, help="Single image path for CLI mode")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    if args.image:
        if not args.run:
            raise ValueError("--run is required when using --image CLI mode")
        run_cli(args.image, args.run, args.device)
        return

    demo = build_demo(run_dir=args.run, runs_root=args.runs_root, device=args.device)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
