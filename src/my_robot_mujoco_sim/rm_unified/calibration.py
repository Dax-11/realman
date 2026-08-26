from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np


def _as_matrix4(value) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got {matrix.shape}")
    return matrix


@dataclass
class EyeHandCalibration:
    camera_to_robot: np.ndarray

    @classmethod
    def identity(cls) -> "EyeHandCalibration":
        return cls(camera_to_robot=np.eye(4, dtype=float))

    @classmethod
    def from_json(cls, path: str) -> "EyeHandCalibration":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        matrix = data.get("camera_to_robot")
        if matrix is None:
            raise KeyError("Missing `camera_to_robot` in calibration file")
        return cls(camera_to_robot=_as_matrix4(matrix))

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"camera_to_robot": self.camera_to_robot.tolist()},
                f,
                indent=2,
                ensure_ascii=False,
            )

    def camera_point_to_robot(self, point_xyz: np.ndarray) -> np.ndarray:
        point_xyz = np.asarray(point_xyz, dtype=float).reshape(3)
        homo = np.ones(4, dtype=float)
        homo[:3] = point_xyz
        mapped = self.camera_to_robot @ homo
        return mapped[:3]
