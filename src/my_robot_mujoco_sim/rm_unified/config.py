from __future__ import annotations

from dataclasses import dataclass
import os


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DEFAULT_SIM_XML = os.path.join(ROOT_DIR, "RM75_Project", "sim_scene.xml")
DEFAULT_REAL_IP = "192.168.1.18"
DEFAULT_D435_CALIBRATION = os.path.join(ROOT_DIR, "configs", "d435_eye_hand_sample.json")


@dataclass
class SimConfig:
    xml_path: str = DEFAULT_SIM_XML
    render: bool = os.environ.get("DISPLAY") is not None
    keep_window_open: bool = True
    timestep: float = 0.002
    settle_steps: int = 200
    control_hz: float = 100.0
    max_joint_step: float = 0.01


@dataclass
class RealArmConfig:
    ip: str = DEFAULT_REAL_IP
    port: int = 8080
    level: int = 3
    move_speed: int = 20
    gripper_open: int = 1000
    gripper_timeout_s: int = 10


@dataclass
class D435VisionConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    center_x: int = 320
    center_y: int = 240
    min_depth_m: float = 0.20
    max_depth_m: float = 1.20
    alpha: float = 0.2
    default_target_x: float = 0.45
    default_target_y: float = 0.0
    default_target_z: float = 0.50
    calibration_path: str = DEFAULT_D435_CALIBRATION
