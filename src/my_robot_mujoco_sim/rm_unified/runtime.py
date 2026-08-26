from __future__ import annotations

import time
from typing import Callable

from .contracts import RobotCommand, RobotState, VisionObservation


PolicyFn = Callable[[RobotState, VisionObservation], RobotCommand | None]


class TaskRunner:
    def __init__(self, robot, vision, policy: PolicyFn, control_hz: float = 30.0):
        self.robot = robot
        self.vision = vision
        self.policy = policy
        self.control_hz = float(control_hz)

    def run(self, duration_s: float) -> None:
        dt = 1.0 / max(self.control_hz, 1e-6)
        end_time = time.time() + duration_s

        while time.time() < end_time:
            state = self.robot.get_state()
            observation = self.vision.get_observation()
            command = self.policy(state, observation)
            if command is not None:
                self.robot.send_command(command)
            time.sleep(dt)
