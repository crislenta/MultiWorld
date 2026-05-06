"""Single-GPU inference helper for the MultiWorld Gradio UI.

The reference inference script (`ittakestwo/parallel_inference.py`) is hard-wired
to torchrun + DDP. For a UI we only need a single GPU and a single sample at a
time, so this module wraps the underlying `WanVideoPipeline` directly without
`torch.distributed`.

Loading the model is expensive so we cache a singleton in `_PIPE`.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

# Make sure the repo root is on sys.path when this module is imported via gradio.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# Model / config loading
# ---------------------------------------------------------------------------

@dataclass
class ModelDescriptor:
    """One row of the model selector dropdown in the UI."""

    name: str
    config_path: str
    checkpoint_path: str
    family: str  # "ittakestwo" or "robots"


# Default catalog. Resolution and config paths follow the readme's `Inference`
# section. Checkpoints are expected to be under ./checkpoints/ inside the repo
# root (the helper scripts in scripts/download_models.sh place them there).
DEFAULT_MODELS: List[ModelDescriptor] = [
    ModelDescriptor(
        name="It Takes Two — 480p (full)",
        config_path="ittakestwo/configs/inference_480P_full.yaml",
        checkpoint_path="checkpoints/multiworld_480p_fulldata.safetensors",
        family="ittakestwo",
    ),
    ModelDescriptor(
        name="It Takes Two — 480p (toy)",
        config_path="ittakestwo/configs/inference_480P_toy.yaml",
        checkpoint_path="checkpoints/multiworld_480p_toydata.safetensors",
        family="ittakestwo",
    ),
    ModelDescriptor(
        name="Robots — 320p",
        config_path="robots/configs/inference.yaml",
        checkpoint_path="checkpoints/multiworld_320p_robots.safetensors",
        family="robots",
    ),
]


def list_available_models() -> List[ModelDescriptor]:
    """Filter `DEFAULT_MODELS` to entries whose checkpoint+config exist on disk."""
    available: List[ModelDescriptor] = []
    for m in DEFAULT_MODELS:
        cfg = _REPO_ROOT / m.config_path
        ckpt = _REPO_ROOT / m.checkpoint_path
        if cfg.exists() and ckpt.exists():
            available.append(m)
    return available


# ---------------------------------------------------------------------------
# Pipeline singleton
# ---------------------------------------------------------------------------

_PIPE = None
_PIPE_KEY: Optional[Tuple[str, str]] = None
_PIPE_LOCK = threading.Lock()


def _load_pipeline(model: ModelDescriptor, device: str = "cuda:0"):
    """Instantiate the WanVideoPipeline for the chosen family and load weights."""
    from utils import load_config  # repo root utils package

    if model.family == "ittakestwo":
        from diffsynth.pipelines.wan_video_ittakestwo import (
            WanVideoPipeline, ModelConfig,
        )
    elif model.family == "robots":
        from diffsynth.pipelines.wan_video_robots import (
            WanVideoPipeline, ModelConfig,
        )
    else:
        raise ValueError(f"Unknown model family: {model.family}")

    cfg = load_config(str(_REPO_ROOT / model.config_path))

    # Patch for Wan2.2 VAE config (matches parallel_inference.py logic).
    if "vae_config" not in cfg.simulator_config:
        vae_model_id = "Wan-AI/Wan2.2-TI2V-5B"
        vae_origin_pattern = "Wan2.2_VAE.pth"
    else:
        vae_model_id = cfg.simulator_config.vae_config.model_id
        vae_origin_pattern = cfg.simulator_config.vae_config.origin_file_pattern

    pipe = WanVideoPipeline.from_pretrained(
        config=cfg,
        torch_dtype=torch.bfloat16,
        device=torch.device(device),
        model_configs=[
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
            ),
            ModelConfig(
                model_id=vae_model_id,
                origin_file_pattern=vae_origin_pattern,
            ),
        ],
    )

    # Load the user-trained checkpoint on top of the Wan2.2 base.
    pretrained = [
        str(_REPO_ROOT / "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors"),
        str(_REPO_ROOT / "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors"),
        str(_REPO_ROOT / "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"),
        str(_REPO_ROOT / model.checkpoint_path),
    ]
    pipe.load_from_checkpoint(pretrained)

    if pipe.env_encoder is not None:
        pipe.env_encoder.to(torch.device(device))

    return pipe, cfg


def get_pipeline(model: ModelDescriptor, device: str = "cuda:0"):
    """Get (or build) the cached pipeline for `model`."""
    global _PIPE, _PIPE_KEY
    key = (model.config_path, model.checkpoint_path)
    with _PIPE_LOCK:
        if _PIPE is None or _PIPE_KEY != key:
            # Drop the previous pipe before allocating the new one so we don't
            # hold two copies of multi-GB weights at once.
            _PIPE = None
            torch.cuda.empty_cache()
            _PIPE, cfg = _load_pipeline(model, device=device)
            _PIPE_KEY = key
            _PIPE.__cfg = cfg  # stash for later access
        return _PIPE, _PIPE.__cfg


# ---------------------------------------------------------------------------
# env_obv preparation (left + right view)
# ---------------------------------------------------------------------------

def _prepare_env_obv(image_path: str, device: torch.device) -> torch.Tensor:
    """Build env_obv tensor `[B=1, F=1, K=2, C, H, W]` from a single image.

    Mirrors `IttakestwoImageActionDataset.load_env_obv_image`.
    """
    from diffsynth.models.vggt.utils.load_fn import load_and_preprocess_images

    left = load_and_preprocess_images([image_path], mode="pad", return_view="left")[None, None, ...]
    right = load_and_preprocess_images([image_path], mode="pad", return_view="right")[None, None, ...]
    env_obv = torch.cat([left, right], dim=2)  # [1, 1, 2, C, H, W]
    return env_obv.to(device=device, dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# Public inference entry point
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    video_path: str
    width: int
    height: int
    num_frames: int


def run_inference(
    model: ModelDescriptor,
    image: Image.Image,
    action: Dict[str, torch.Tensor],
    *,
    seed: int = 0,
    num_inference_steps: int = 35,
    num_frames: int = 81,
    output_dir: str = "outputs/ui",
    device: str = "cuda:0",
    view: Optional[str] = None,
) -> InferenceResult:
    """Run a single inference and write the result mp4 to disk.

    `view` overrides the dataset's `return_view` setting. If left as None we
    fall back to the value declared in the eval config (typically 'random' or
    'both').
    """
    from diffsynth.utils.data import save_video

    pipe, cfg = get_pipeline(model, device=device)
    torch_device = torch.device(device)

    # ---- height / width / frames ----
    dataset_cfg = cfg.eval_dataset_config.params
    height = int(dataset_cfg.video_params.height)
    width_full = int(dataset_cfg.video_params.width)
    return_view = view or dataset_cfg.get("return_view", "random")
    if return_view == "both":
        width = width_full
    else:
        # The dataset stores side-by-side stereo; a single view is half-width.
        width = width_full // 2

    num_frames = int(num_frames)

    # ---- env_obv from the user-supplied image ----
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        image.convert("RGB").save(tmp_path)
    try:
        env_obv = _prepare_env_obv(tmp_path, torch_device)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # ---- Move action tensors to device ----
    action = {
        k: v.to(torch_device,
                dtype=torch.bfloat16 if v.is_floating_point() else torch.long)
        for k, v in action.items()
    }

    # ---- Crop the input image to the chosen view to match the model ----
    img = image.convert("RGB")
    if return_view == "left":
        img = img.crop((0, 0, img.width // 2, img.height))
    elif return_view == "right":
        img = img.crop((img.width // 2, 0, img.width, img.height))

    # ---- Generate ----
    frames = pipe(
        input_image=img,
        action=action,
        env_obv=env_obv,
        seed=int(seed),
        tiled=False,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=int(num_inference_steps),
    )

    # ---- Save ----
    fps = int(60 // dataset_cfg.video_params.frame_skip)
    out_name = f"ui_seed{seed}_steps{num_inference_steps}_f{num_frames}.mp4"
    out_path = str(Path(output_dir) / out_name)
    ffmpeg_params = [
        "-vcodec", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]
    save_video(frames, out_path, fps=fps, quality=10, ffmpeg_params=ffmpeg_params)

    return InferenceResult(
        video_path=out_path,
        width=width,
        height=height,
        num_frames=num_frames,
    )
