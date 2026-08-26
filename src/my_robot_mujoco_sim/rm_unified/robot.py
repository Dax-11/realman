from __future__ import annotations

import os
import time
import gc

import numpy as np

from .config import RealArmConfig, SimConfig
from .contracts import RobotCommand, RobotState


class RobotBackend:
    def reset(self) -> None:
        raise NotImplementedError

    def send_command(self, command: RobotCommand) -> None:
        raise NotImplementedError

    def get_state(self) -> RobotState:
        raise NotImplementedError

    def close(self) -> None:
        return None

    @property
    def dof(self) -> int:
        raise NotImplementedError


class SimRobotBackend(RobotBackend):
    def __init__(self, config: SimConfig | None = None):
        self.config = config or SimConfig()

        try:
            import mujoco
            import mujoco.viewer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing `mujoco` Python package. Install via `pip install mujoco`."
            ) from exc

        if not os.path.exists(self.config.xml_path):
            raise FileNotFoundError(f"MuJoCo scene file not found: {self.config.xml_path}")

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(self.config.xml_path)
        self.data = mujoco.MjData(self.model)

        self._dof = self.model.nu if self.model.nu > 0 else self.model.nq
        self._ee_id = self._resolve_site_id("end_effector")

        self._viewer = None
        self._render_enabled = bool(self.config.render)
        self._gripper = 0.0

        if self._render_enabled:
            try:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception as exc:
                self._render_enabled = False
                print(f"[Sim] Viewer launch failed: {exc}")

        self._ctrl_home = np.zeros(self.model.nu, dtype=float)
        self.reset()

        print(
            f"[Sim] Loaded: {os.path.basename(self.config.xml_path)}  DOF={self._dof}  "
            f"render={'on' if self._render_enabled else 'off'}"
        )

    @property
    def dof(self) -> int:
        return self._dof

    def reset(self) -> None:
        try:
            self._mj.mj_resetDataKeyframe(self.model, self.data, 0)
        except Exception:
            self._mj.mj_resetData(self.model, self.data)

        self._mj.mj_forward(self.model, self.data)
        self._ctrl_home[:self._dof] = np.asarray(self.data.qpos[:self._dof], dtype=float)

        if self.model.nu > 0:
            self.data.ctrl[:self._dof] = self._ctrl_home[:self._dof]

        if self.model.nu > self._dof:
            self.data.ctrl[self._dof:] = 0.0

        self._step_and_sync(5)

    def send_command(self, command: RobotCommand) -> None:
        if command.joint_pos is not None:
            target = self._normalize_joint_target(command.joint_pos)
            self._move_joints_safely(target)

        if command.gripper is not None:
            self._gripper = float(np.clip(command.gripper, 0.0, 1.0))
            if self.model.nu > self._dof:
                self.data.ctrl[self._dof:] = self._gripper
                self._step_and_sync(20)

    def get_state(self) -> RobotState:
        self._mj.mj_forward(self.model, self.data)
        ee_pos = np.array(self.data.site_xpos[self._ee_id]) if self._ee_id is not None else np.zeros(3)

        return RobotState(
            joint_pos=np.asarray(self.data.qpos[:self._dof], dtype=float).copy(),
            ee_pos=ee_pos.copy(),
            gripper=self._gripper,
            sim_time=float(self.data.time),
            raw={"ctrl": np.asarray(self.data.ctrl[:self.model.nu], dtype=float).copy()},
        )

    def close(self) -> None:
        viewer = self._viewer
        self._viewer = None
        if viewer is not None:
            try:
                if self.config.keep_window_open and viewer.is_running():
                    print("[Sim] Simulation finished, MuJoCo window kept open.")
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(0.03)
                viewer.close()
                time.sleep(0.1)
            except Exception:
                pass
        gc.collect()

    def _resolve_site_id(self, site_name: str) -> int | None:
        try:
            return self.model.site(site_name).id
        except Exception:
            return None

    def _normalize_joint_target(self, target: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=float).reshape(-1)
        if len(target) < self._dof:
            target = np.concatenate([target, np.zeros(self._dof - len(target))])
        elif len(target) > self._dof:
            target = target[:self._dof]
        return target

    def _move_joints_safely(self, target: np.ndarray) -> None:
        self._mj.mj_forward(self.model, self.data)
        current = np.asarray(self.data.qpos[:self._dof], dtype=float).copy()

        error = target - current
        max_error = float(np.max(np.abs(error))) if len(error) else 0.0

        n_ctrl_steps = max(1, int(np.ceil(max_error / max(self.config.max_joint_step, 1e-4))))

        for alpha in np.linspace(0.0, 1.0, n_ctrl_steps + 1)[1:]:
            desired = current + alpha * error

            if self.model.nu > 0:
                self.data.ctrl[:self._dof] = desired
            else:
                self.data.qpos[:self._dof] = desired
                self._mj.mj_forward(self.model, self.data)

            self._step_and_sync(3)

        if self.model.nu > 0:
            self.data.ctrl[:self._dof] = target
        self._step_and_sync(self.config.settle_steps)

    def _step_and_sync(self, n_steps: int) -> None:
        for _ in range(max(1, int(n_steps))):
            self._mj.mj_step(self.model, self.data)

            if not np.all(np.isfinite(self.data.qpos[:self._dof])):
                raise RuntimeError("MuJoCo simulation state contains NaN/Inf.")

        if self._viewer and self._viewer.is_running():
            self._viewer.sync()


class RealRobotBackend(RobotBackend):
    def __init__(self, config: RealArmConfig | None = None):
        self.config = config or RealArmConfig()

        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "Python"))
        from Robotic_Arm.rm_robot_interface import (
            RM_MOVE_MULTI_BLOCK,
            RoboticArm,
            rm_thread_mode_e,
            rm_trajectory_connect_config_e,
        )

        self._rm_traj = rm_trajectory_connect_config_e
        self._block = RM_MOVE_MULTI_BLOCK

        self._rm = RoboticArm(rm_thread_mode_e(2))

        handle = self._rm.rm_create_robot_arm(self.config.ip, self.config.port, self.config.level)
        if handle.id == -1:
            raise ConnectionError(
                f"Failed to connect robot arm at {self.config.ip}:{self.config.port}."
            )

        info = self._rm.rm_get_robot_info()
        self._dof = info[1].get("arm_dof", 7)
        self._gripper = 0.0
        print(f"[Real] Connected {self.config.ip}:{self.config.port}  DOF={self._dof}")

        import threading
        self._sdk_lock = threading.Lock()
        self._state_cache = None

        self._force_raw_read()

        self._poll_running = True
        self._poller_thread = threading.Thread(target=self._sdk_poller_loop, daemon=True)
        self._poller_thread.start()

    @property
    def dof(self) -> int:
        return self._dof

    def reset(self) -> None:
        self._movej(np.zeros(self._dof, dtype=float), speed=self.config.move_speed)

    def send_command(self, command: RobotCommand) -> None:
        if command.joint_pos is not None:
            speed = max(1, min(100, int(command.speed_ratio * 100)))
            target = np.asarray(command.joint_pos, dtype=float).reshape(-1)
            if len(target) < self._dof:
                target = np.concatenate([target, np.zeros(self._dof - len(target))])
            self._movej(target[:self._dof], speed=speed)

        if command.gripper is not None:
            self._gripper = float(np.clip(command.gripper, 0.0, 1.0))
            position = int(self._gripper * self.config.gripper_open)
            with self._sdk_lock:
                tag = self._rm.rm_set_gripper_position(
                    position,
                    block=True,
                    timeout=self.config.gripper_timeout_s,
                )
            if tag != 0:
                raise RuntimeError(f"Gripper control failed, error code: {tag}")

    def _force_raw_read(self) -> None:
        import time
        with self._sdk_lock:
            joint_ret = self._rm.rm_get_joint_degree()
            pose_ret = self._rm.rm_get_current_arm_state()

        joint_rad = np.deg2rad(np.asarray(joint_ret[1][:self._dof], dtype=float))
        ee_pos = np.asarray(pose_ret[1]["pose"][:3], dtype=float) / 1000.0

        self._state_cache = RobotState(
            joint_pos=joint_rad,
            ee_pos=ee_pos,
            gripper=self._gripper,
            sim_time=time.time(),
            raw={
                "joint_ret": joint_ret[0],
                "pose_ret": pose_ret[0],
                "joint_deg": list(joint_ret[1][:self._dof]),
                "pose_raw": list(pose_ret[1]["pose"]),
            },
        )

    def _sdk_poller_loop(self):
        import time
        while self._poll_running:
            try:
                self._force_raw_read()
                time.sleep(0.002)
            except Exception:
                time.sleep(0.01)

    def get_state(self) -> RobotState:
        return self._state_cache

    def close(self) -> None:
        import os
        self._poll_running = False
        if hasattr(self, '_poller_thread') and self._poller_thread.is_alive():
            self._poller_thread.join(timeout=1.0)
        with self._sdk_lock:
            self._rm.rm_delete_robot_arm()
        print("[Real] Disconnected")

    def _movej(self, angles_rad: np.ndarray, speed: int) -> None:
        target_deg = np.rad2deg(np.asarray(angles_rad, dtype=float)).tolist()
        with self._sdk_lock:
            tag = self._rm.rm_movej(
                target_deg,
                speed,
                0,
                self._rm_traj.RM_TRAJECTORY_DISCONNECT_E,
                self._block,
            )
        if tag != 0:
            raise RuntimeError(f"movej failed, error code: {tag}")


class RSRobotBackend(RobotBackend):
    def __init__(self, **kwargs):
        import threading
        import time
        import os

        print("\n[RS] Starting digital twin engine...")

        real_config = kwargs.get("real_config")
        self._real_backend = RealRobotBackend(real_config)

        base_sim_config = kwargs.get("sim_config") or SimConfig()

        self._temp_xml = self._build_ghost_xml(base_sim_config.xml_path)
        sim_config = SimConfig(xml_path=self._temp_xml, render=True, keep_window_open=base_sim_config.keep_window_open)

        self._sim_backend = SimRobotBackend(sim_config)

        self._ghost_qpos_adrs = []
        for i in range(self._sim_backend.model.njnt):
            name = self._sim_backend._mj.mj_id2name(self._sim_backend.model, self._sim_backend._mj.mjtObj.mjOBJ_JOINT, i)
            if name and name.endswith('_ghost'):
                self._ghost_qpos_adrs.append(self._sim_backend.model.jnt_qposadr[i])

        self._voxel_mocap_ids = []
        for i in range(self._sim_backend.model.nbody):
            name = self._sim_backend._mj.mj_id2name(self._sim_backend.model, self._sim_backend._mj.mjtObj.mjOBJ_BODY, i)
            if name and name.startswith('voxel_'):
                mocap_id = self._sim_backend.model.body_mocapid[i]
                if mocap_id >= 0:
                    self._voxel_mocap_ids.append(mocap_id)

        self._target_qpos = None

        self._latency_comp_ms = kwargs.get("latency_comp_ms", 22.0)
        self._qvel_smoothed = None
        self._ema_alpha = 0.7

        self._use_d435 = kwargs.get("use_d435_voxel", True)
        self._rs_pipeline = None
        if self._use_d435:
            try:
                import pyrealsense2 as rs
                self._rs_pipeline = rs.pipeline()
                cfg = rs.config()
                cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                self._rs_pipeline.start(cfg)
                self._pc = rs.pointcloud()
                print("[RS] D435 stream enabled.")
            except Exception as e:
                print(f"[RS] D435 device missing or pyrealsense2 error: {e}")
                self._rs_pipeline = None

        self._running = True
        self._shadow_thread = threading.Thread(target=self._shadow_sync_loop, daemon=True)
        self._shadow_thread.start()

    def _build_ghost_xml(self, base_xml):
        import xml.etree.ElementTree as ET
        import os
        tree = ET.parse(base_xml)
        root = tree.getroot()
        worldbody = root.find('worldbody')

        if worldbody is not None:
            for body in list(worldbody.findall('body')):
                ghost_body = ET.fromstring(ET.tostring(body, encoding='unicode'))
                for elem in ghost_body.iter():
                    if 'name' in elem.attrib:
                        elem.attrib['name'] += '_ghost'
                    if elem.tag == 'geom':
                        elem.attrib['rgba'] = '0 1 0 0.35'
                        elem.attrib['contype'] = '0'
                        elem.attrib['conaffinity'] = '0'
                worldbody.append(ghost_body)

            for i in range(1024):
                voxel = ET.Element('body', name=f'voxel_{i}', mocap='true')
                ET.SubElement(voxel, 'geom', type='box', size='0.005 0.005 0.005',
                              rgba='0 0.8 1 0.7', contype='0', conaffinity='0', pos='0 0 -10')
                worldbody.append(voxel)

        ghost_path = os.path.join(os.path.dirname(base_xml), "rs_ghost_twin.xml")
        tree.write(ghost_path)
        return ghost_path

    def _shadow_sync_loop(self):
        import time
        import numpy as np
        last_time = time.time()
        last_qpos = None

        while self._running:
            try:
                if not getattr(self._sim_backend, '_viewer', None) or not self._sim_backend._viewer.is_running():
                    time.sleep(0.5)
                    continue

                real_state = self._real_backend.get_state()
                current_qpos = real_state.joint_pos.copy()
                current_time = time.time()
                dt = max(current_time - last_time, 1e-4)

                if last_qpos is not None:
                    raw_qvel = (current_qpos - last_qpos) / dt

                    if self._qvel_smoothed is None:
                        self._qvel_smoothed = raw_qvel
                    else:
                        self._qvel_smoothed = self._ema_alpha * self._qvel_smoothed + (1.0 - self._ema_alpha) * raw_qvel

                    self._sim_backend.data.qvel[:self.dof] = self._qvel_smoothed

                    if self._latency_comp_ms > 0:
                        predict_dt = self._latency_comp_ms / 1000.0
                        predicted_qpos = current_qpos + self._qvel_smoothed * predict_dt
                        self._sim_backend.data.qpos[:self.dof] = predicted_qpos
                    else:
                        self._sim_backend.data.qpos[:self.dof] = current_qpos
                else:
                    self._sim_backend.data.qpos[:self.dof] = current_qpos

                last_time = current_time
                last_qpos = current_qpos

                if self._target_qpos is not None:
                    for i, adr in enumerate(self._ghost_qpos_adrs):
                        if i < len(self._target_qpos):
                            self._sim_backend.data.qpos[adr] = self._target_qpos[i]

                if getattr(self, '_rs_pipeline', None) is not None:
                    frames = self._rs_pipeline.poll_for_frames()
                    if frames:
                        depth_frame = frames.get_depth_frame()
                        if depth_frame:
                            points = self._pc.calculate(depth_frame)
                            vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

                            valid_idx = np.where(vtx[:, 2] > 0.1)[0]
                            sampled_len = 0
                            if len(valid_idx) > 0:
                                samples = np.random.choice(valid_idx, min(len(valid_idx), len(self._voxel_mocap_ids)), replace=False)
                                sampled_pts = vtx[samples]
                                sampled_len = len(sampled_pts)

                                for i, mocap_id in enumerate(self._voxel_mocap_ids):
                                    if i < sampled_len:
                                        pt = sampled_pts[i]
                                        self._sim_backend.data.mocap_pos[mocap_id] = [pt[0], pt[2] + 0.3, -pt[1] + 0.3]

                            for i in range(sampled_len, len(self._voxel_mocap_ids)):
                                self._sim_backend.data.mocap_pos[self._voxel_mocap_ids[i]] = [0, 0, -10]

                self._sim_backend._mj.mj_forward(self._sim_backend.model, self._sim_backend.data)
                self._sim_backend._viewer.sync()

                time.sleep(0.016)

            except Exception:
                time.sleep(0.1)

    @property
    def dof(self) -> int:
        return self._real_backend.dof

    def reset(self) -> None:
        self._real_backend.reset()

    def send_command(self, command: RobotCommand) -> None:
        if command.joint_pos is not None:
            self._target_qpos = np.asarray(command.joint_pos, dtype=float).copy()
        self._real_backend.send_command(command)

    def get_state(self) -> RobotState:
        return self._real_backend.get_state()

    def close(self) -> None:
        import os
        print("[RS] Closing digital twin...")
        self._running = False
        if hasattr(self, '_shadow_thread') and self._shadow_thread.is_alive():
            self._shadow_thread.join(timeout=1.0)
        self._real_backend.close()
        self._sim_backend.close()

        if getattr(self, '_rs_pipeline', None) is not None:
            try:
                self._rs_pipeline.stop()
            except Exception: pass

        if hasattr(self, '_temp_xml') and os.path.exists(self._temp_xml):
            try:
                os.remove(self._temp_xml)
            except Exception: pass


class UnifiedRobot:
    def __init__(self, backend: str = "sim", **kwargs):
        if backend == "sim":
            config = kwargs.get("config") or SimConfig(
                xml_path=kwargs.get("xml_path", SimConfig().xml_path),
                render=kwargs.get("render", SimConfig().render),
                keep_window_open=kwargs.get("keep_window_open", SimConfig().keep_window_open),
            )
            self._backend = SimRobotBackend(config)

        elif backend == "real":
            config = kwargs.get("config") or RealArmConfig(
                ip=kwargs.get("ip", RealArmConfig().ip),
                port=kwargs.get("port", RealArmConfig().port),
                level=kwargs.get("level", RealArmConfig().level),
            )
            self._backend = RealRobotBackend(config)

        elif backend == "rs":
            self._backend = RSRobotBackend(**kwargs)

        else:
            raise ValueError(f"backend must be 'sim', 'real', or 'rs', got: {backend!r}")

        self.backend_name = backend

    @property
    def dof(self) -> int:
        return self._backend.dof

    def reset(self) -> None:
        self._backend.reset()

    def send_command(self, command: RobotCommand) -> None:
        self._backend.send_command(command)

    def set_joint_pos(self, angles, speed_ratio: float = 0.2) -> None:
        self.send_command(RobotCommand(joint_pos=np.asarray(angles, dtype=float), speed_ratio=speed_ratio))

    def set_gripper(self, value: float) -> None:
        self.send_command(RobotCommand(gripper=float(value)))

    def get_state(self) -> RobotState:
        return self._backend.get_state()

    def get_joint_pos(self) -> np.ndarray:
        return self.get_state().joint_pos

    def get_ee_pos(self) -> np.ndarray:
        return self.get_state().ee_pos

    def close(self) -> None:
        self._backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
