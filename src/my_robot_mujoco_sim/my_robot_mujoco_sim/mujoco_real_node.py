#!/usr/bin/env python3
import os
import importlib.util
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

import mujoco
import mujoco.viewer

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Empty, Float64MultiArray, Int32, String
from std_srvs.srv import Trigger


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _probe_tcp(host: str, port: int, timeout_s: float = 1.0) -> bool:
    """快速 TCP 端口探测，避免 SDK 长时间阻塞在握手"""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


class MuJoCoRealBridgeNode(Node):

    JOINT_NAMES = [
        'joint1', 'joint2', 'joint3',
        'joint4', 'joint5', 'joint6', 'joint7'
    ]

    def __init__(self):
        super().__init__('mujoco_real_bridge_node')

        # ---------- 1. 参数 ----------
        self.declare_parameter('scene_file',     'sim_scene.xml')
        self.declare_parameter('camera_name',    'd435_view')
        self.declare_parameter('headless',       False)
        self.declare_parameter('move_duration',  3.0)

        # 真臂相关
        self.declare_parameter('enable_real',    True)
        self.declare_parameter('real_ip',        '192.168.1.18')
        self.declare_parameter('real_port',      8080)
        self.declare_parameter('real_level',     3)
        self.declare_parameter('stream_rate_hz', 50.0)   # CANFD 透传频率
        self.declare_parameter('follow_high',    True)   # 高跟随
        self.declare_parameter('trajectory_mode', 1)     # 1=曲线拟合，更稳
        self.declare_parameter('radio',          50)     # 平滑系数
        self.declare_parameter('min_move_duration', 2.0)
        self.declare_parameter('motion_start_hold_time', 0.2)
        self.declare_parameter('send_idle_hold_position', False)
        self.declare_parameter('publish_real_state', True)
        self.declare_parameter('enable_gripper', False)
        self.declare_parameter(
            'gripper_module_path',
            '/home/kan/ros2_wss/gripper_modbus_final.py')
        self.declare_parameter('gripper_device_id', 1)
        self.declare_parameter('gripper_baudrate', 115200)
        self.declare_parameter('gripper_open_position', 0)
        self.declare_parameter('gripper_close_position', 9000)
        self.declare_parameter('gripper_position_max', 9000)
        self.declare_parameter('gripper_closed_input_value', 0.85)
        self.declare_parameter('gripper_default_speed', 50)
        self.declare_parameter('gripper_default_force', 30)
        self.declare_parameter('gripper_release_force', 100)
        self.declare_parameter('gripper_command_timeout', 5.0)
        self.declare_parameter('gripper_poll_interval', 0.05)
        self.declare_parameter('gripper_position_deadband', 15)

        # LISTEN 模式：订阅外部 /joint_states（例如 rm_driver 或 MoveIt）
        # 让 MuJoCo 做"只读可视化镜像"，不触碰真臂
        self.declare_parameter('listen_mode',        False)
        self.declare_parameter('listen_topic',       '/joint_states')

        scene_file  = self.get_parameter('scene_file').value
        self._cam   = self.get_parameter('camera_name').value
        headless    = _as_bool(self.get_parameter('headless').value)
        self._has_display = bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
        self._mujoco_gl = os.environ.get('MUJOCO_GL', '').strip().lower()
        self._can_offscreen_render = self._has_display or self._mujoco_gl in ('egl', 'osmesa')
        self._traj_default_dur = self.get_parameter('move_duration').value

        self._enable_real = _as_bool(self.get_parameter('enable_real').value)
        self._real_ip     = str(self.get_parameter('real_ip').value)
        self._real_port   = int(self.get_parameter('real_port').value)
        self._real_level  = int(self.get_parameter('real_level').value)
        self._stream_dt   = 1.0 / max(1.0, float(self.get_parameter('stream_rate_hz').value))
        self._follow_high = _as_bool(self.get_parameter('follow_high').value)
        self._traj_mode   = int(self.get_parameter('trajectory_mode').value)
        self._radio       = int(self.get_parameter('radio').value)
        self._min_move_duration = float(self.get_parameter('min_move_duration').value)
        self._motion_start_hold_time = float(self.get_parameter('motion_start_hold_time').value)
        self._send_idle_hold_position = _as_bool(self.get_parameter('send_idle_hold_position').value)
        self._pub_real    = _as_bool(self.get_parameter('publish_real_state').value)
        self._enable_gripper = _as_bool(self.get_parameter('enable_gripper').value)
        self._gripper_module_path = Path(str(self.get_parameter('gripper_module_path').value))
        self._gripper_device_id = int(self.get_parameter('gripper_device_id').value)
        self._gripper_baudrate = int(self.get_parameter('gripper_baudrate').value)
        self._gripper_open_position = int(self.get_parameter('gripper_open_position').value)
        self._gripper_close_position = int(self.get_parameter('gripper_close_position').value)
        self._gripper_position_max = int(self.get_parameter('gripper_position_max').value)
        self._gripper_closed_input_value = float(self.get_parameter('gripper_closed_input_value').value)
        self._gripper_default_speed = int(self.get_parameter('gripper_default_speed').value)
        self._gripper_default_force = int(self.get_parameter('gripper_default_force').value)
        self._gripper_release_force = int(self.get_parameter('gripper_release_force').value)
        self._gripper_command_timeout = float(self.get_parameter('gripper_command_timeout').value)
        self._gripper_poll_interval = float(self.get_parameter('gripper_poll_interval').value)
        self._gripper_position_deadband = int(self.get_parameter('gripper_position_deadband').value)
        self._listen_mode = _as_bool(self.get_parameter('listen_mode').value)
        self._listen_topic = str(self.get_parameter('listen_topic').value)

        # ---------- 2. 加载 MuJoCo ----------
        pkg_share  = get_package_share_directory('my_robot_mujoco_sim')
        scene_path = os.path.join(pkg_share, 'models', scene_file)
        mesh_dir   = os.path.join(pkg_share, 'models', 'RM75_6F')
        self.get_logger().info(f'场景文件: {scene_path}')

        assets = self._load_assets(mesh_dir)
        try:
            if assets:
                self.model = mujoco.MjModel.from_xml_path(scene_path, assets=assets)
            else:
                self.model = mujoco.MjModel.from_xml_path(scene_path)
        except Exception as e:
            self.get_logger().error(f'模型加载失败: {e}')
            raise
        self.data = mujoco.MjData(self.model)

        joint1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
        self.qpos_start = self.model.jnt_qposadr[joint1_id] if joint1_id >= 0 else 0
        self.dof_start = self.model.jnt_dofadr[joint1_id] if joint1_id >= 0 else 0

        self._apply_home_keyframe()
        self._randomize_block_position()

        # 检查相机是否存在
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self._cam)
        self._has_camera = (cam_id != -1)
        if not self._has_camera:
            self.get_logger().warn(f"相机 '{self._cam}' 不存在")

        self._lock = threading.Lock()

        # 平滑轨迹状态
        self._traj_start_pos  = None
        self._traj_target_pos = None
        self._traj_start_time = None
        self._traj_duration   = self._traj_default_dur
        self._traj_hold_time  = 0.0
        self._traj_active     = False  # 标志：轨迹运动是否进行中
        self._traj_gripper_target = None
        self._last_move_signature = None
        self._last_move_wall_time = 0.0
        self._duplicate_move_window_s = 30.0

        # ---------- 3. 真臂连接（带探测） ----------
        self._real_arm = None
        self._real_lock = threading.Lock()
        self._real_dof = 7
        self._real_state_cache = None  # (qpos_rad)
        self._mode = 'sim_only'
        self._real_gripper = None
        self._gripper_lock = threading.Lock()
        self._last_gripper_target = None
        self._last_gripper_command_time = 0.0
        self._last_gripper_result = {}
        self._last_gripper_init_error = None

        # LISTEN 模式时不碰真臂，只订阅 /joint_states 做镜像
        if self._listen_mode:
            self._mode = 'listen'
            self.get_logger().info(
                f"LISTEN 模式：订阅 {self._listen_topic}，MuJoCo 只做可视化镜像")
        elif self._enable_real:
            self._try_connect_real_arm()

        self._publish_mode()

        # ---------- 4. ROS 通信 ----------
        # LISTEN 模式不发布 /joint_states（让 rm_driver 的保持唯一），改发布 /mujoco/joint_states_mirror
        joint_state_topic = '/mujoco/joint_states_mirror' if self._listen_mode else '/joint_states'
        self.joint_pub = self.create_publisher(JointState, joint_state_topic, 10)
        self.mode_pub  = self.create_publisher(String, '/mujoco/mode', 10)
        self.image_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.gripper_status_pub = self.create_publisher(String, '/ctag2f90d/status', 10)
        self._last_image_pub_time = 0.0

        # 固定俯视相机发布（table_cam，始终对准桌面）
        self._table_cam_name = 'table_cam'
        table_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self._table_cam_name)
        self._has_table_cam = (table_cam_id != -1)
        if self._has_table_cam:
            self.get_logger().info(f"固定俯视相机 '{self._table_cam_name}' 已找到，将发布至 /camera/color/image_raw")
        else:
            self.get_logger().warn(f"固定俯视相机 '{self._table_cam_name}' 未找到，回退到 '{self._cam}'")

        self.create_subscription(Float64MultiArray, '/joint_commands',
                                 self._cmd_callback, 10)
        self.create_subscription(Float64MultiArray, '/joint_move_to',
                                 self._move_to_callback, 10)
        self.create_subscription(Empty, '/ctag2f90d/open',
                                 self._ctag_open_callback, 10)
        self.create_subscription(Empty, '/ctag2f90d/close',
                                 self._ctag_close_callback, 10)
        self.create_subscription(Int32, '/ctag2f90d/target_position',
                                 self._ctag_target_position_callback, 10)
        self.create_service(Trigger, '/mujoco/reset', self._reset_callback)

        if self._enable_gripper and self._real_arm is not None:
            self._try_init_real_gripper()
        elif self._enable_gripper:
            self.get_logger().warn('已请求启用真实夹爪，但当前没有真臂连接，跳过夹爪初始化')
        else:
            self.get_logger().info('真实夹爪未启用；如需启用请启动时添加 enable_gripper:=true')

        # LISTEN 模式：订阅外部 /joint_states 驱动 MuJoCo 可视化
        if self._listen_mode:
            self.create_subscription(
                JointState, self._listen_topic,
                self._listen_joint_state_cb, 10)

        # ---------- 5. 仿真步进 ----------
        self._step_count = 0
        dt = float(self.model.opt.timestep)
        self.create_timer(dt, self._sim_step)
        self.create_timer(1.0, self._publish_gripper_debug_status)

        # ---------- 6. 真臂流式跟随线程 ----------
        self._stream_running = False
        self._stream_thread = None
        if self._real_arm is not None:
            self._stream_running = True
            # 把仿真的状态传到真机
            self._stream_thread = threading.Thread(
                target=self._real_stream_loop, daemon=True, name='real_stream')
            self._stream_thread.start()

            self._real_poll_running = True
            self._real_poll_thread = threading.Thread(
                target=self._real_poll_loop, daemon=True, name='real_poll')
            self._real_poll_thread.start()

        # ---------- 6.5. 图像发布 ----------
        self._cam_pub_running = False
        self._cam_pub_thread = None
        if headless and self._can_offscreen_render:
            self._cam_pub_running = True
            self._cam_pub_thread = threading.Thread(
                target=self._camera_publish_loop, daemon=True, name='camera_pub')
            self._cam_pub_thread.start()
        elif headless:
            self.get_logger().warn(
                '无头模式下未检测到 DISPLAY/WAYLAND_DISPLAY，且 MUJOCO_GL 不是 egl/osmesa；'
                '跳过离屏相机发布')

        # ---------- 7. 渲染线程 ----------
        if not headless and self._has_display:
            t = threading.Thread(target=self._render_loop, daemon=True, name='mujoco_render')
            t.start()
        elif not headless:
            self.get_logger().warn('未检测到 DISPLAY/WAYLAND_DISPLAY，跳过 MuJoCo GUI 窗口')
        else:
            self.get_logger().info('无头模式，不启动渲染窗口')

        self.get_logger().info(
            f'启动！模式={self._mode}  timestep={dt*1000:.1f}ms  '
            f'nu={self.model.nu}  nq={self.model.nq}'
        )

    # 真臂连接管理
    def _try_connect_real_arm(self):
        """探测 + 连接真臂，失败回退到 sim_only"""
        if not _probe_tcp(self._real_ip, self._real_port, timeout_s=1.0):
            self.get_logger().warn(
                f'未检测到真臂 {self._real_ip}:{self._real_port}，回退到纯仿真')
            self._mode = 'sim_only'
            return

        try:
            # Robotic_Arm 包已通过 ament_python_install_package 装到 site-packages
            from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

            self.get_logger().info(f'尝试连接真臂 {self._real_ip}:{self._real_port}...')
            arm = RoboticArm(rm_thread_mode_e(2))
            handle = arm.rm_create_robot_arm(self._real_ip, self._real_port, self._real_level)
            if handle is None or getattr(handle, 'id', -1) == -1:
                self.get_logger().warn('真臂连接失败（handle=-1），回退到纯仿真')
                self._mode = 'sim_only'
                return

            info = arm.rm_get_robot_info()
            if info and len(info) >= 2 and isinstance(info[1], dict):
                self._real_dof = info[1].get('arm_dof', 7)

            # 切到透传模式：mode=4 表示开放透传/CANFD
            try:
                arm.rm_set_arm_run_mode(4)
            except Exception:
                pass

            # 用真臂当前位姿初始化 MuJoCo，避免开窗瞬间猛跳
            joint_ret = arm.rm_get_joint_degree()
            if joint_ret and joint_ret[0] == 0:
                qdeg = list(joint_ret[1][:self._real_dof])
                qrad = np.deg2rad(qdeg)
                with self._lock:
                    self.data.qpos[self.qpos_start : self.qpos_start + self._real_dof] = qrad
                    if self.model.nu > 0:
                        self.data.ctrl[:self._real_dof] = qrad
                    mujoco.mj_forward(self.model, self.data)
                self._real_state_cache = qrad

            self._real_arm = arm
            self._mode = 'mirror'
            self.get_logger().info(
                f'已连接真臂 ({self._real_ip})  DOF={self._real_dof}  -> MIRROR 模式')
        except Exception as e:
            self.get_logger().warn(f'真臂初始化异常 ({type(e).__name__}: {e})，回退到纯仿真')
            self._mode = 'sim_only'
            self._real_arm = None

    def _publish_mode(self):
        if hasattr(self, 'mode_pub'):
            msg = String()
            msg.data = self._mode
            self.mode_pub.publish(msg)

    def _try_init_real_gripper(self):
        self._last_gripper_init_error = None
        if not self._gripper_module_path.exists():
            self._last_gripper_init_error = f'找不到夹爪控制文件: {self._gripper_module_path}'
            self.get_logger().error(self._last_gripper_init_error)
            return
        try:
            spec = importlib.util.spec_from_file_location(
                'gripper_modbus_final', self._gripper_module_path)
            if spec is None or spec.loader is None:
                self._last_gripper_init_error = f'无法加载夹爪控制文件: {self._gripper_module_path}'
                self.get_logger().error(self._last_gripper_init_error)
                return
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            self._real_gripper = module.GripperModbusController(
                self._real_arm,
                device_id=self._gripper_device_id,
                baudrate=self._gripper_baudrate,
            )
            with self._real_lock:
                if not self._real_gripper.switch_to_modbus_rtu_mode():
                    self._real_gripper = None
                    self._last_gripper_init_error = '切换工具端 RS485 到 Modbus RTU 主站失败'
                    self.get_logger().error(self._last_gripper_init_error)
                    return
                ret = self._real_gripper.enable()
                if ret != 0:
                    self._real_gripper = None
                    self._last_gripper_init_error = f'夹爪电机使能失败: {ret}'
                    self.get_logger().error(self._last_gripper_init_error)
                    return
                self._real_gripper.reset_alarm()

            self.get_logger().info(
                '真实 CTAG2F90D 夹爪已接入 /joint_move_to 第8个值，'
                '格式: [q1..q7, gripper, duration]')
        except Exception as e:
            self._real_gripper = None
            self._last_gripper_init_error = f'真实夹爪初始化失败: {type(e).__name__}: {e}'
            self.get_logger().error(self._last_gripper_init_error)

    def _ensure_real_gripper_initialized(self) -> bool:
        if self._real_gripper is not None:
            return True
        if not self._enable_gripper:
            self._last_gripper_init_error = 'enable_gripper 参数为 false'
            return False
        if self._real_arm is None:
            self._last_gripper_init_error = '真臂未连接，无法初始化夹爪'
            return False
        self.get_logger().warn('真实夹爪尚未初始化，收到命令后尝试重新初始化...')
        self._try_init_real_gripper()
        return self._real_gripper is not None

    def _map_gripper_value_to_position(self, value: float) -> int:
        denom = self._gripper_closed_input_value \
            if abs(self._gripper_closed_input_value) > 1e-6 else 1.0
        ratio = max(0.0, min(1.0, value / denom))
        position = self._gripper_open_position + ratio * (
            self._gripper_close_position - self._gripper_open_position)
        return int(round(max(0, min(self._gripper_position_max, position))))

    def _dispatch_real_gripper_command(self, gripper_value: float):
        if not self._ensure_real_gripper_initialized():
            self._warn_throttle(
                '收到夹爪目标，但真实夹爪未初始化。请确认 enable_gripper:=true，'
                '“真实 CTAG2F90D 夹爪已接入”。')
            return
        target_position = self._map_gripper_value_to_position(gripper_value)
        self.get_logger().info(
            f'收到 /joint_move_to 夹爪值 {gripper_value:.3f} -> CTAG2F90D 目标位置 {target_position}')
        self._dispatch_real_gripper_position(target_position)

    def _dispatch_real_gripper_position(self, target_position: int):
        if not self._ensure_real_gripper_initialized():
            self._warn_throttle(
                '收到直接夹爪位置命令，但真实夹爪未初始化。请确认 enable_gripper:=true。')
            return
        target_position = max(0, min(self._gripper_position_max, int(target_position)))
        now = time.time()
        if (
            self._last_gripper_target is not None
            and abs(target_position - self._last_gripper_target) <= self._gripper_position_deadband
            and now - self._last_gripper_command_time < 0.25
        ):
            return

        self._last_gripper_target = target_position
        self._last_gripper_command_time = now
        threading.Thread(
            target=self._run_real_gripper_command,
            args=(target_position,),
            daemon=True,
            name='real_gripper_cmd',
        ).start()

    def _ctag_open_callback(self, _msg: Empty):
        self.get_logger().info(f'收到 /ctag2f90d/open -> 位置 {self._gripper_open_position}')
        self._dispatch_real_gripper_position(self._gripper_open_position)

    def _ctag_close_callback(self, _msg: Empty):
        self.get_logger().info(f'收到 /ctag2f90d/close -> 位置 {self._gripper_close_position}')
        self._dispatch_real_gripper_position(self._gripper_close_position)

    def _ctag_target_position_callback(self, msg: Int32):
        self.get_logger().info(f'收到 /ctag2f90d/target_position -> 位置 {msg.data}')
        self._dispatch_real_gripper_position(msg.data)

    def _run_real_gripper_command(self, target_position: int):
        if not self._gripper_lock.acquire(blocking=False):
            self._warn_throttle('上一条夹爪命令仍在执行，忽略新的夹爪目标')
            return
        try:
            force = self._gripper_release_force \
                if target_position <= self._gripper_open_position + self._gripper_position_deadband \
                else self._gripper_default_force
            with self._real_lock:
                ret = self._real_gripper.move(
                    position=target_position,
                    speed=self._gripper_default_speed,
                    force=force,
                )
            if ret != 0:
                result = {
                    'completed': False,
                    'grip_success': False,
                    'send_error': ret,
                    'target_position': target_position,
                }
            else:
                result = self._wait_real_gripper_complete(target_position)
            self._last_gripper_result = result
            self.gripper_status_pub.publish(
                String(data=json.dumps(result, ensure_ascii=False)))
            self.get_logger().info(f'真实夹爪命令完成: {result}')
        except Exception as e:
            result = {
                'completed': False,
                'grip_success': False,
                'error': f'{type(e).__name__}: {e}',
                'target_position': target_position,
            }
            self._last_gripper_result = result
            self.gripper_status_pub.publish(
                String(data=json.dumps(result, ensure_ascii=False)))
            self.get_logger().error(f'真实夹爪命令失败: {result}')
        finally:
            self._gripper_lock.release()

    def _wait_real_gripper_complete(self, target_position: int) -> dict:
        start = time.time()
        force_reached_first = False
        position_reached_first = False
        final_state = {}
        final_position = None

        while time.time() - start < self._gripper_command_timeout:
            with self._real_lock:
                final_state = self._real_gripper.get_motion_state()
                final_position = self._real_gripper.get_real_position()

            if final_state:
                if final_state.get('force_reached') and not position_reached_first:
                    force_reached_first = True
                if final_state.get('position_reached') and not force_reached_first:
                    position_reached_first = True
                if final_state.get('zero_speed_reached') and final_state.get('ready'):
                    return {
                        'completed': True,
                        'grip_success': force_reached_first and not position_reached_first,
                        'final_state': final_state,
                        'final_position': final_position,
                        'target_position': target_position,
                        'elapsed': time.time() - start,
                    }
            time.sleep(self._gripper_poll_interval)

        return {
            'completed': False,
            'grip_success': False,
            'final_state': final_state,
            'final_position': final_position,
            'target_position': target_position,
            'elapsed': self._gripper_command_timeout,
        }

    def _publish_gripper_debug_status(self):
        if not hasattr(self, 'gripper_status_pub'):
            return
        payload = {
            'enabled_param': self._enable_gripper,
            'initialized': self._real_gripper is not None,
            'mode': self._mode,
            'last_target': self._last_gripper_target,
            'last_result': self._last_gripper_result,
            'init_error': self._last_gripper_init_error,
        }
        self.gripper_status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def _warn_throttle(self, text: str):
        now = time.time()
        last = getattr(self, '_warn_last', 0.0)
        if now - last > 5.0:
            self._warn_last = now
            self.get_logger().warn(text)

    # MuJoCo 资源

    def _load_assets(self, mesh_dir: str) -> dict:
        assets = {}
        if not os.path.exists(mesh_dir):
            return assets
        for fname in os.listdir(mesh_dir):
            if fname.lower().endswith('.stl'):
                with open(os.path.join(mesh_dir, fname), 'rb') as f:
                    assets[fname] = f.read()
        if assets:
            self.get_logger().info(f'加载了 {len(assets)} 个 STL 资源')
        return assets

    def _apply_home_keyframe(self):
        if self.model.nkey <= 0:
            return
        home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if home_id == -1:
            self.get_logger().warn("找不到 'home' 关键帧")
            return
        self.data.qpos[:] = self.model.key_qpos[home_id]
        if self.model.nu > 0:
            self.data.ctrl[:] = self.model.key_ctrl[home_id]
        mujoco.mj_forward(self.model, self.data)
        self.get_logger().info("已应用 home 关键帧")


    # 仿真步进 + 轨迹插值

    def _sim_step(self):
        with self._lock:
            self._update_trajectory()
            mujoco.mj_step(self.model, self.data)
        
        self._step_count += 1
        if self._step_count % 5 == 0:
            self._publish_joint_states()

    def _update_trajectory(self):
        if self._traj_target_pos is None:
            return
        elapsed = max(0.0, self.data.time - self._traj_start_time - self._traj_hold_time)
        alpha   = min(elapsed / self._traj_duration, 1.0)
        s = alpha ** 3 * (alpha * (alpha * 6 - 15) + 10)
        self.data.ctrl[:7] = (
            self._traj_start_pos
            + s * (self._traj_target_pos - self._traj_start_pos)
        )
        if alpha >= 1.0:
            self._traj_target_pos = None
            self._traj_active = False
            self.get_logger().info('运动完成')

    # 真臂流式跟随：把 MuJoCo 的 ctrl 透传给真臂
    def _real_stream_loop(self):
        last_target_deg = None
        idle_skip_threshold_deg = 0.05  # 小于此变化不发送，省带宽

        while self._stream_running and rclpy.ok():
            t0 = time.time()
            try:
                # 仅在轨迹活跃时持续流，否则按需发送
                with self._lock:
                    target = self.data.ctrl[:7].copy() if self.model.nu >= 7 \
                             else self.data.qpos[self.qpos_start : self.qpos_start + 7].copy()
                target_deg = np.rad2deg(target).tolist()

                send = False
                if self._traj_active:
                    send = True
                elif not self._send_idle_hold_position:
                    send = False
                elif last_target_deg is None:
                    send = True
                else:
                    diff = max(abs(a - b) for a, b in zip(target_deg, last_target_deg))
                    if diff > idle_skip_threshold_deg:
                        send = True

                if send and self._real_arm is not None:
                    with self._real_lock:
                        try:
                            self._real_arm.rm_movej_canfd(
                                target_deg,
                                follow=self._follow_high,
                                expand=0,
                                trajectory_mode=self._traj_mode,
                                radio=self._radio,
                            )
                        except Exception as e:
                            self._warn_throttle(
                                f'真臂透传异常: {type(e).__name__}: {e}')
                    last_target_deg = target_deg
            except Exception as e:
                self._warn_throttle(f'流循环异常: {e}')

            elapsed = time.time() - t0
            sleep_t = max(0.0, self._stream_dt - elapsed)
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _real_poll_loop(self):

        while self._real_poll_running and rclpy.ok():
            try:
                with self._real_lock:
                    ret = self._real_arm.rm_get_joint_degree()
                if ret and ret[0] == 0:
                    qrad = np.deg2rad(np.asarray(ret[1][:self._real_dof], dtype=float))
                    self._real_state_cache = qrad
            except Exception:
                time.sleep(0.05)
            time.sleep(0.01)  # 100Hz

    def _get_live_real_joint_position(self):
        if self._mode != 'mirror' or self._real_arm is None:
            return None
        try:
            with self._real_lock:
                ret = self._real_arm.rm_get_joint_degree()
            if ret and ret[0] == 0:
                qrad = np.deg2rad(np.asarray(ret[1][:self._real_dof], dtype=float))
                if len(qrad) >= 7:
                    self._real_state_cache = qrad[:7].copy()
                    return qrad[:7].copy()
        except Exception as exc:
            self._warn_throttle(f'读取真臂起点失败，改用仿真起点: {type(exc).__name__}: {exc}')
        if self._real_state_cache is not None and len(self._real_state_cache) >= 7:
            return self._real_state_cache[:7].copy()
        return None

    # 渲染与图像发布（独立图像发布线程，即使窗口关闭或处于无头模式也会持续工作）
    def _camera_publish_loop(self):
        renderer = None
        try:
            renderer = mujoco.Renderer(self.model, height=480, width=640)
        except Exception as e:
            self.get_logger().error(f"Failed to initialize: {e}")
            return

        self.get_logger().info("独立图像发布线程启动")
        while self._cam_pub_running and rclpy.ok():
            start_time = time.time()
            if renderer is not None:
                try:
                    with self._lock:
                        if self._has_table_cam:
                            renderer.update_scene(self.data, camera=self._table_cam_name)
                        elif self._has_camera:
                            renderer.update_scene(self.data, camera=self._cam)
                        else:
                            continue
                        image = renderer.render()
                    self._publish_image(image)
                except Exception as e:
                    self.get_logger().error(f"Error in camera: {e}")
            
            elapsed = time.time() - start_time
            sleep_time = max(0.001, 0.05 - elapsed)
            time.sleep(sleep_time)

    def _render_loop(self):
        renderer = None
        try:
            renderer = mujoco.Renderer(self.model, height=480, width=640)
        except Exception as e:
            self.get_logger().error(f"Failed to initialize : {e}")

        viewer = None
        try:
            with self._lock:
                viewer = mujoco.viewer.launch_passive(
                    self.model, self.data,
                    show_left_ui=True, show_right_ui=True)
            self.get_logger().info('MuJoCo 窗口已打开')
            while viewer.is_running():
                with self._lock:
                    viewer.sync()
                    if renderer is not None:
                        now = time.time()
                        if now - self._last_image_pub_time >= 0.05:
                            if self._has_table_cam:
                                renderer.update_scene(self.data, camera=self._table_cam_name)
                            elif self._has_camera:
                                renderer.update_scene(self.data, camera=self._cam)
                            else:
                                continue
                            image = renderer.render()
                            self._last_image_pub_time = now
                        else:
                            image = None
                    else:
                        image = None

                if image is not None:
                    self._publish_image(image)
                threading.Event().wait(timeout=0.01)
        finally:
            if renderer is not None:
                renderer.close()
            if viewer is not None:
                viewer.close()
        self.get_logger().info('MuJoCo 窗口已关闭')

    # ROS 回调
    def _cmd_callback(self, msg: Float64MultiArray):
        if self._listen_mode:
            self.get_logger().warn_once = None  # 占位避免 warn_once 缺失
            self._warn_throttle('LISTEN 模式下忽略 /joint_commands（由上游驱动）')
            return
        if len(msg.data) != 7:
            self.get_logger().warn(f'期望7个值，收到{len(msg.data)}')
            return
        with self._lock:
            self._traj_target_pos = None
            self._traj_active = False
            self.data.ctrl[:7] = np.array(msg.data)

    def _move_to_callback(self, msg: Float64MultiArray):
        if self._listen_mode:
            self._warn_throttle('LISTEN 模式下忽略 /joint_move_to（由上游驱动）')
            return
        n = len(msg.data)
        if n not in (7, 8, 9):
            self.get_logger().warn(f'期望7、8或9个值，收到{n}')
            return
        target = np.array(msg.data[:7])
        gripper_target = None
        if n == 9:
            gripper_target = float(msg.data[7])
            duration = float(msg.data[8])
        elif n == 8:
            duration = float(msg.data[7])
        else:
            duration = self._traj_default_dur

        live_start = self._get_live_real_joint_position()

        with self._lock:
            duration = max(self._min_move_duration, duration)
            signature = (
                tuple(np.round(target, 6).tolist()),
                None if gripper_target is None else round(gripper_target, 6),
                round(duration, 6),
            )
            now = time.time()
            duplicate_recent_command = (
                signature == self._last_move_signature
                and now - self._last_move_wall_time < self._duplicate_move_window_s
            )
            if duplicate_recent_command:
                return

            if live_start is not None:
                start_pos = live_start
                self.data.qpos[self.qpos_start : self.qpos_start + 7] = start_pos
                self.data.qvel[self.dof_start : self.dof_start + 7] = 0.0
                if self.model.nu >= 7:
                    self.data.ctrl[:7] = start_pos
                mujoco.mj_forward(self.model, self.data)
            else:
                start_pos = self.data.qpos[self.qpos_start : self.qpos_start + 7].copy()

            self._traj_start_pos  = start_pos.copy()
            self._traj_target_pos = target
            self._traj_start_time = self.data.time
            self._traj_duration   = duration
            self._traj_hold_time  = max(0.0, self._motion_start_hold_time)
            self._traj_active     = True
            self._traj_gripper_target = gripper_target
            self._last_move_signature = signature
            self._last_move_wall_time = now
            if gripper_target is not None and self.model.nu > 7:
                self.data.ctrl[7] = gripper_target
        self.get_logger().info(
            f'[{self._mode}] → 目标: {np.round(target,3).tolist()} | 夹爪: {gripper_target} | 时长={duration}s')
        if gripper_target is not None:
            self._dispatch_real_gripper_command(gripper_target)

    def _reset_callback(self, request, response):
        with self._lock:
            self._traj_target_pos = None
            self._traj_active = False
            self._traj_gripper_target = None
            self._last_move_signature = None
            self._last_move_wall_time = 0.0
            if not self._listen_mode:
                self._apply_home_keyframe()
                self._randomize_block_position()
        response.success = True
        response.message = f'已重置 (mode={self._mode})'
        return response

    def _listen_joint_state_cb(self, msg: JointState):
        """LISTEN 模式：把外部 /joint_states 回灌到 MuJoCo qpos"""
        if not msg.position:
            return
        # 按名字顺序匹配，兼容 rm_driver、joint_state_broadcaster
        positions = {}
        for n, p in zip(msg.name, msg.position):
            positions[n] = p
        qs = []
        for jn in self.JOINT_NAMES:
            if jn in positions:
                qs.append(positions[jn])
        if len(qs) != 7:
            return
        with self._lock:
            self.data.qpos[self.qpos_start : self.qpos_start + 7] = np.array(qs)
            if self.model.nu >= 7:
                self.data.ctrl[:7] = np.array(qs)
            mujoco.mj_forward(self.model, self.data)

    def _publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.JOINT_NAMES

        if (self._mode == 'mirror' and self._pub_real and
                self._real_state_cache is not None):
            qrad = self._real_state_cache
            msg.position = qrad.tolist() + [0.0] * (7 - len(qrad)) if len(qrad) < 7 else qrad[:7].tolist()
            msg.velocity = []
            msg.effort   = []
        else:
            msg.position = self.data.qpos[self.qpos_start : self.qpos_start + 7].tolist()
            msg.velocity = self.data.qvel[self.dof_start : self.dof_start + 7].tolist()
            msg.effort   = self.data.actuator_force[:7].tolist()

        self.joint_pub.publish(msg)
        self._publish_mode()

    def _publish_image(self, rgb_image):
        if rgb_image is None:
            return
        try:
            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_color_optical_frame"
            msg.height = rgb_image.shape[0]
            msg.width = rgb_image.shape[1]
            msg.encoding = "rgb8"
            msg.is_bigendian = 0
            msg.step = rgb_image.shape[1] * 3
            msg.data = rgb_image.tobytes()
            self.image_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing image: {e}")

    def _randomize_block_position(self):
        try:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "block_joint")
            if joint_id != -1:
                qpos_adr = self.model.jnt_qposadr[joint_id]
                
                random_x = np.random.uniform(-0.36, -0.28)
                random_y = np.random.uniform(-0.08, 0.08)
                random_z = 0.42
                
                yaw = np.random.uniform(-np.pi/6, np.pi/6) # -30 to 30 degrees
                qw = np.cos(yaw / 2.0)
                qz = np.sin(yaw / 2.0)
                
                self.data.qpos[qpos_adr : qpos_adr + 3] = [random_x, random_y, random_z]
                self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = [qw, 0.0, 0.0, qz]
                mujoco.mj_forward(self.model, self.data)
                self.get_logger().info(f"已随机化木块位置: X={random_x:.3f}, Y={random_y:.3f}, Yaw={np.degrees(yaw):.1f}°")
        except Exception as e:
            self.get_logger().error(f"随机化木块位置失败: {e}")

    # 清理
    def destroy_node(self):
        try:
            self._stream_running = False
            self._real_poll_running = False
            self._cam_pub_running = False
            if self._real_arm is not None:
                with self._real_lock:
                    try:
                        self._real_arm.rm_delete_robot_arm()
                    except Exception:
                        pass
                self._real_arm = None
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MuJoCoRealBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
