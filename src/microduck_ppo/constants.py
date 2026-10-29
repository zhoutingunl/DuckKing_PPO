"""Shared MicroDuck policy interface copied from the DuckKing controller contract."""

from __future__ import annotations

import numpy as np

POLICY_JOINTS = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)

HOME = np.array(
    [
        0,
        -0.0873,
        -0.4579,
        -0.0049,
        0.453,
        0.3491,
        0.3491,
        0,
        0,
        0,
        0.0873,
        0.4579,
        0.0049,
        -0.453,
    ],
    dtype=np.float32,
)

# The original DuckKing scene runs at 200 Hz and infers its policy at 50 Hz.
PHYSICS_TIMESTEP = 0.005
CONTROL_INTERVAL = 0.02
ACTION_SCALE = 0.20
