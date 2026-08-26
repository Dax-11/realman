from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RobotState:
    joint_pos: np.ndarray
    ee_pos: np.ndarray
    gripper: float = 0.0
    sim_time: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotCommand:
    joint_pos: np.ndarray | None = None
    gripper: float | None = None
    speed_ratio: float = 0.2


@dataclass
class VisionObservation:
    target_pos: np.ndarray | None = None
    target_visible: bool = False
    raw: dict[str, Any] = field(default_factory=dict)
