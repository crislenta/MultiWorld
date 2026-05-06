"""Video / image I/O helpers for the diffsynth pipelines.

This module restores the `save_video` symbol expected by the inference scripts
(`ittakestwo/parallel_inference.py`, `robots/parallel_inference.py`, and the
internal `diffsynth/diffusion/runner.py`). The upstream commit ships a
reference to `diffsynth.utils.data` but the file itself was never committed,
so any inference run currently fails at import time with
``ModuleNotFoundError: No module named 'diffsynth.utils.data'``.

The implementation is a thin imageio wrapper that accepts the
``List[PIL.Image]`` produced by ``BasePipeline.vae_output_to_video`` (the
default ``output_type="quantized"`` path) as well as the raw ``np.ndarray``
frames that some scripts hand it directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
from PIL import Image

try:
    import imageio.v2 as imageio
except ImportError:  # imageio<2.10 fallback
    import imageio  # type: ignore


Frame = Union[Image.Image, np.ndarray]


def _frame_to_uint8(frame: Frame) -> np.ndarray:
    """Convert a single frame (PIL or ndarray) to an HxWx3 uint8 RGB array."""
    if isinstance(frame, Image.Image):
        return np.asarray(frame.convert("RGB"))

    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(
            f"save_video expected HxWx{{3,4}} frames, got shape {arr.shape}"
        )
    if arr.shape[-1] == 4:  # drop alpha
        arr = arr[..., :3]
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        # Assume range [0, 1] or [0, 255] and clip.
        if arr.max() <= 1.0 + 1e-3:
            arr = arr * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)
    return arr.astype(np.uint8)


def save_video(
    frames: Iterable[Frame],
    save_path: Union[str, Path],
    fps: int = 24,
    quality: int = 9,
    ffmpeg_params: Optional[Sequence[str]] = None,
) -> str:
    """Write `frames` to `save_path` as an MP4.

    Args:
        frames: Iterable of `PIL.Image` or `HxWx3` `np.ndarray` (uint8 or float
            in [0, 1]). Lists from `BasePipeline.vae_output_to_video` work
            directly.
        save_path: Output path (.mp4). Parent directories are created if
            missing.
        fps: Frame rate of the encoded video.
        quality: imageio quality (0–10, higher is better). Ignored if
            `ffmpeg_params` overrides codec/crf.
        ffmpeg_params: Extra args forwarded to the ffmpeg backend. Use this to
            pin codec / crf / pixfmt (the canonical preset used by the
            inference scripts is shown in the module example below).

    Example:
        >>> save_video(frames, "out.mp4", fps=30, quality=10, ffmpeg_params=[
        ...     "-vcodec", "libx264", "-preset", "medium", "-crf", "18",
        ...     "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        ... ])
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    writer_kwargs = {"fps": fps}
    if quality is not None:
        writer_kwargs["quality"] = quality
    if ffmpeg_params:
        writer_kwargs["ffmpeg_params"] = list(ffmpeg_params)

    writer = imageio.get_writer(str(save_path), **writer_kwargs)
    try:
        for frame in frames:
            writer.append_data(_frame_to_uint8(frame))
    finally:
        writer.close()

    return str(save_path)


def save_frames(frames: Iterable[Frame], save_dir: Union[str, Path], prefix: str = "frame", ext: str = "png") -> List[str]:
    """Save each frame as an individual image. Useful for debugging."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for i, frame in enumerate(frames):
        arr = _frame_to_uint8(frame)
        out = save_dir / f"{prefix}_{i:06d}.{ext}"
        Image.fromarray(arr).save(out)
        paths.append(str(out))
    return paths


__all__ = ["save_video", "save_frames"]
