from __future__ import annotations

import time
from typing import Iterable, Sequence

import numpy as np

from .contracts import RobotCommand, RobotState


class JointSequenceTask:
    def __init__(self, waypoints: Iterable[Sequence[float]], speed_ratio: float = 0.2, pause_s: float = 0.5):
        self._waypoints = [np.asarray(waypoint, dtype=float) for waypoint in waypoints]
        self._speed_ratio = float(speed_ratio)
        self._pause_s = float(pause_s)

    def run(self, robot) -> list[RobotState]:
        states: list[RobotState] = []
        for waypoint in self._waypoints:
            robot.send_command(RobotCommand(joint_pos=waypoint, speed_ratio=self._speed_ratio))
            states.append(robot.get_state())
            if self._pause_s > 0:
                time.sleep(self._pause_s)
        return states
