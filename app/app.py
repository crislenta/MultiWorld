"""Gradio UI for testing the MultiWorld video world model.

Run:
    python -m app.app                 # default 0.0.0.0:7860
    python -m app.app --share         # public Gradio link
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional

import gradio as gr
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.action_builder import DISCRETE_KEYS, build_action_from_ui  # noqa: E402
from app.inference import (  # noqa: E402
    DEFAULT_MODELS,
    ModelDescriptor,
    list_available_models,
    run_inference,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXAMPLE_IMAGES_DIR = _REPO_ROOT / "assets"


def _model_choices() -> List[str]:
    avail = list_available_models()
    if avail:
        return [m.name for m in avail]
    # Show the full list anyway so the UI is informative — generation will fail
    # with a clear error if the user picks one whose checkpoint is missing.
    return [m.name for m in DEFAULT_MODELS]


def _resolve_model(name: str) -> ModelDescriptor:
    for m in DEFAULT_MODELS:
        if m.name == name:
            return m
    raise ValueError(f"Unknown model: {name}")


def _example_image_paths() -> List[str]:
    if not EXAMPLE_IMAGES_DIR.exists():
        return []
    return [
        str(p) for p in sorted(EXAMPLE_IMAGES_DIR.iterdir())
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]


# ---------------------------------------------------------------------------
# Generation callback
# ---------------------------------------------------------------------------

def generate(
    model_name: str,
    image: Optional[Image.Image],
    view: str,
    num_frames: int,
    num_inference_steps: int,
    seed: int,
    left_keys: List[str],
    left_look_x: float,
    left_look_y: float,
    right_keys: List[str],
    right_look_x: float,
    right_look_y: float,
    progress=gr.Progress(track_tqdm=True),
):
    if image is None:
        raise gr.Error("Please upload an initial frame first.")

    try:
        model = _resolve_model(model_name)
    except ValueError as e:
        raise gr.Error(str(e))

    # Robots family doesn't have a meaningful left/right player split — we
    # still feed the same shape but the UI labels are slightly misleading.
    action = build_action_from_ui(
        num_frames=int(num_frames),
        left_keys=left_keys,
        left_look_x=float(left_look_x),
        left_look_y=float(left_look_y),
        right_keys=right_keys,
        right_look_x=float(right_look_x),
        right_look_y=float(right_look_y),
    )

    progress(0.05, desc="Loading pipeline (first call may take a few minutes)")

    try:
        result = run_inference(
            model=model,
            image=image,
            action=action,
            seed=int(seed),
            num_inference_steps=int(num_inference_steps),
            num_frames=int(num_frames),
            view=None if view == "auto" else view,
        )
    except FileNotFoundError as e:
        raise gr.Error(
            "A required checkpoint or config is missing on disk. "
            "Run `bash scripts/download_models.sh` first.\n\n"
            f"Original error: {e}"
        )
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Inference failed: {e}")

    info = (
        f"Generated {result.num_frames} frames at "
        f"{result.width}x{result.height}\n"
        f"Saved to: {result.video_path}"
    )
    return result.video_path, info


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

CSS = """
#title h1 { margin-bottom: 0.2em; }
#title p  { margin-top: 0; color: var(--body-text-color-subdued); }
.action-card { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 12px; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="MultiWorld Tester", css=CSS, theme=gr.themes.Soft()) as demo:
        gr.HTML(
            """
            <div id="title">
                <h1>MultiWorld — Interactive Tester</h1>
                <p>Upload an initial frame, dial in a constant action for each player,
                and generate a short rollout from the multi-agent multi-view video
                world model.</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                model_dd = gr.Dropdown(
                    choices=_model_choices(),
                    value=(_model_choices()[0] if _model_choices() else None),
                    label="Model checkpoint",
                    interactive=True,
                )

                image_in = gr.Image(
                    label="Initial frame (side-by-side stereo expected for It Takes Two)",
                    type="pil",
                    height=260,
                )

                examples = _example_image_paths()
                if examples:
                    gr.Examples(
                        examples=[[p] for p in examples],
                        inputs=[image_in],
                        label="Example frames from ./assets",
                    )

                view_dd = gr.Dropdown(
                    choices=["auto", "left", "right", "both"],
                    value="auto",
                    label="View",
                    info="'auto' uses the value from the config. "
                         "'both' generates the full stereo width.",
                )

                with gr.Row():
                    num_frames = gr.Slider(
                        13, 81, value=81, step=4,
                        label="Number of frames",
                        info="Must satisfy time_division_factor=4 + remainder=1.",
                    )
                    num_steps = gr.Slider(
                        10, 80, value=35, step=1, label="Inference steps",
                    )
                    seed = gr.Number(value=0, precision=0, label="Seed")

            with gr.Column(scale=1):
                with gr.Group(elem_classes="action-card"):
                    gr.Markdown("### Player 1 (left)")
                    left_keys = gr.CheckboxGroup(
                        choices=DISCRETE_KEYS,
                        value=[],
                        label="Buttons held",
                    )
                    with gr.Row():
                        left_lx = gr.Slider(-1.0, 1.0, value=0.0, step=0.05,
                                            label="Look X (norm_dx)")
                        left_ly = gr.Slider(-1.0, 1.0, value=0.0, step=0.05,
                                            label="Look Y (norm_dy)")

                with gr.Group(elem_classes="action-card"):
                    gr.Markdown("### Player 2 (right)")
                    right_keys = gr.CheckboxGroup(
                        choices=DISCRETE_KEYS,
                        value=[],
                        label="Buttons held",
                    )
                    with gr.Row():
                        right_lx = gr.Slider(-1.0, 1.0, value=0.0, step=0.05,
                                             label="Look X")
                        right_ly = gr.Slider(-1.0, 1.0, value=0.0, step=0.05,
                                             label="Look Y")

                run_btn = gr.Button("Generate video", variant="primary")

        with gr.Row():
            video_out = gr.Video(label="Generated rollout", height=360)
            info_out = gr.Textbox(label="Run info", lines=4, interactive=False)

        run_btn.click(
            generate,
            inputs=[
                model_dd, image_in, view_dd, num_frames, num_steps, seed,
                left_keys, left_lx, left_ly,
                right_keys, right_lx, right_ly,
            ],
            outputs=[video_out, info_out],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("GRADIO_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_PORT", "7860")))
    parser.add_argument("--share", action="store_true",
                        help="Create a public *.gradio.live tunnel (handy on a "
                             "fresh GCP VM with no firewall rules yet).")
    args = parser.parse_args()

    demo = build_ui()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_api=False,
    )


if __name__ == "__main__":
    main()
