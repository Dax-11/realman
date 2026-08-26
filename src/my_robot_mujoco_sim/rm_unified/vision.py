from __future__ import annotations

from dataclasses import dataclass
import os
import numpy as np

from .calibration import EyeHandCalibration
from .config import D435VisionConfig
from .contracts import VisionObservation


class VisionBackend:
    def get_observation(self) -> VisionObservation:
        raise NotImplementedError

    def close(self) -> None:
        return None


@dataclass
class StaticTargetVision(VisionBackend):
    target_pos: tuple[float, float, float] = (0.35, 0.0, 0.35)

    def get_observation(self) -> VisionObservation:
        return VisionObservation(
            target_pos=np.asarray(self.target_pos, dtype=float),
            target_visible=True,
        )


class NoVision(VisionBackend):
    def get_observation(self) -> VisionObservation:
        return VisionObservation(target_pos=None, target_visible=False)


class D435VisionBackend(VisionBackend):
    def __init__(self, config: D435VisionConfig | None = None):
        self.config = config or D435VisionConfig()

        try:
            import pyrealsense2 as rs
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing `pyrealsense2`. Install RealSense bindings before using D435VisionBackend."
            ) from exc

        self._rs = rs

        self.pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(
            rs.stream.depth,
            self.config.width,
            self.config.height,
            rs.format.z16,
            self.config.fps,
        )
        self.pipeline.start(rs_config)

        self._filtered_target = np.array(
            [
                self.config.default_target_x,
                self.config.default_target_y,
                self.config.default_target_z,
            ],
            dtype=float,
        )

        self._calibration = self._load_calibration(self.config.calibration_path)

    def get_observation(self) -> VisionObservation:
        frames = self.pipeline.wait_for_frames()
        depth = frames.get_depth_frame()

        if not depth:
            return VisionObservation(
                target_pos=self._filtered_target.copy(),
                target_visible=False,
                raw={"reason": "no_depth_frame"},
            )

        dist = float(depth.get_distance(self.config.center_x, self.config.center_y))
        visible = self.config.min_depth_m < dist < self.config.max_depth_m

        if visible:
            intrinsics = depth.profile.as_video_stream_profile().intrinsics
            camera_point = self._rs.rs2_deproject_pixel_to_point(
                intrinsics,
                [self.config.center_x, self.config.center_y],
                dist,
            )

            raw_target = self._calibration.camera_point_to_robot(np.asarray(camera_point, dtype=float))

            self._filtered_target = (
                self.config.alpha * raw_target + (1.0 - self.config.alpha) * self._filtered_target
            )

        return VisionObservation(
            target_pos=self._filtered_target.copy(),
            target_visible=visible,
            raw={
                "depth_m": dist,
                "center_px": [self.config.center_x, self.config.center_y],
                "calibration_path": self.config.calibration_path,
            },
        )

    def close(self) -> None:
        self.pipeline.stop()

    def _load_calibration(self, path: str) -> EyeHandCalibration:
        if path and os.path.exists(path):
            return EyeHandCalibration.from_json(path)
        return EyeHandCalibration.identity()
