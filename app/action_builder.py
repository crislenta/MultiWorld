"""Build the (discrete_action, continuous_action) tensor dict expected by the
ItTakesTwo / Robots WanVideoPipeline from simple UI inputs.

Pipeline expects (under `convert_gamepad_to_keyboard=True`):
    discrete_action  : Tensor[B, F, 2, 10]  int64
    continuous_action: Tensor[B, F, 2,  2]  float32 (cast to bf16 by caller)

Player index 0 = left player (keyboard + mouse).
Player index 1 = right player (gamepad mapped onto the same 10-d keyboard space).

Discrete index ordering (10-d): w, a, s, d, space, shift, ctrl, e, q, f
Continuous index ordering (2-d): look_x, look_y
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch


DISCRETE_KEYS: List[str] = [
    "w", "a", "s", "d", "space", "shift", "ctrl", "e", "q", "f",
]
CONTINUOUS_KEYS: List[str] = ["look_x", "look_y"]


@dataclass
class PlayerAction:
    """A single, constant action applied for every frame of the clip."""

    w: bool = False
    a: bool = False
    s: bool = False
    d: bool = False
    space: bool = False
    shift: bool = False
    ctrl: bool = False
    e: bool = False
    q: bool = False
    f: bool = False
    look_x: float = 0.0
    look_y: float = 0.0

    def discrete_vector(self) -> np.ndarray:
        return np.array(
            [int(getattr(self, k)) for k in DISCRETE_KEYS], dtype=np.int64
        )

    def continuous_vector(self) -> np.ndarray:
        return np.array(
            [float(self.look_x), float(self.look_y)], dtype=np.float32
        )


@dataclass
class ActionPlan:
    """Per-player constant actions for a single video clip."""

    num_frames: int = 81
    left: PlayerAction = field(default_factory=PlayerAction)
    right: PlayerAction = field(default_factory=PlayerAction)

    def build(self) -> Dict[str, torch.Tensor]:
        F = int(self.num_frames)

        # [F, 2, 10]
        discrete = np.zeros((F, 2, len(DISCRETE_KEYS)), dtype=np.int64)
        discrete[:, 0, :] = self.left.discrete_vector()[None, :]
        discrete[:, 1, :] = self.right.discrete_vector()[None, :]

        # [F, 2, 2]
        continuous = np.zeros((F, 2, len(CONTINUOUS_KEYS)), dtype=np.float32)
        continuous[:, 0, :] = self.left.continuous_vector()[None, :]
        continuous[:, 1, :] = self.right.continuous_vector()[None, :]

        return {
            "discrete_action": torch.from_numpy(discrete)[None, ...],   # [1, F, 2, 10]
            "continuous_action": torch.from_numpy(continuous)[None, ...],  # [1, F, 2, 2]
        }


def build_action_from_ui(
    num_frames: int,
    left_keys: Optional[List[str]] = None,
    left_look_x: float = 0.0,
    left_look_y: float = 0.0,
    right_keys: Optional[List[str]] = None,
    right_look_x: float = 0.0,
    right_look_y: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Convenience wrapper used by the Gradio UI.

    `left_keys` / `right_keys` are lists of strings drawn from `DISCRETE_KEYS`
    representing the buttons that should be held down for every frame.
    """
    def _to_player(keys: Optional[List[str]], lx: float, ly: float) -> PlayerAction:
        keys = set(keys or [])
        return PlayerAction(
            **{k: (k in keys) for k in DISCRETE_KEYS},
            look_x=lx,
            look_y=ly,
        )

    plan = ActionPlan(
        num_frames=num_frames,
        left=_to_player(left_keys, left_look_x, left_look_y),
        right=_to_player(right_keys, right_look_x, right_look_y),
    )
    return plan.build()
