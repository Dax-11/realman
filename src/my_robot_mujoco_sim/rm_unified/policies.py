from __future__ import annotations

import numpy as np

from .contracts import RobotCommand, RobotState, VisionObservation


class HoldCurrentPosePolicy:
    def __call__(self, state: RobotState, observation: VisionObservation) -> RobotCommand:
        return RobotCommand(joint_pos=state.joint_pos.copy(), speed_ratio=0.05)


class VisionNudgePolicy:
    def __init__(self, step_rad: float = 0.01):
        self.step_rad = float(step_rad)

    def __call__(self, state: RobotState, observation: VisionObservation) -> RobotCommand | None:
        if not observation.target_visible or observation.target_pos is None:
            return None

        joints = state.joint_pos.copy()
        target = observation.target_pos

        if len(joints) >= 1:
            joints[0] = np.clip(joints[0] + np.sign(target[1]) * self.step_rad, -0.5, 0.5)

        if len(joints) >= 2:
            joints[1] = np.clip(joints[1] - np.sign(target[0] - 0.45) * self.step_rad, -0.4, 0.4)

        if len(joints) >= 3:
            joints[2] = np.clip(joints[2] + np.sign(target[2] - 0.50) * self.step_rad, -0.3, 0.3)

        return RobotCommand(joint_pos=joints, speed_ratio=0.05)


class ReachSafePresetPolicy:
    def __call__(self, state: RobotState, observation: VisionObservation) -> RobotCommand:
        joints = state.joint_pos.copy()

        if observation.target_visible and observation.target_pos is not None:
            target = observation.target_pos

            preset = np.array(
                [
                    np.clip(target[1] * 0.6, -0.35, 0.35),
                    np.clip(-0.05 - (target[0] - 0.40) * 0.5, -0.25, 0.20),
                    np.clip(0.05 + (target[2] - 0.45) * 0.5, -0.10, 0.25),
                    0.0,
                    0.04,
                    0.0,
                    0.0,
                ],
                dtype=float,
            )
            joints[: min(len(joints), len(preset))] = preset[: min(len(joints), len(preset))]

        return RobotCommand(joint_pos=joints, speed_ratio=0.05)
