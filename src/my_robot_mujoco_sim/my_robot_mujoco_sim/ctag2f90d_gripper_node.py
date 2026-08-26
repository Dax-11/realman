#!/usr/bin/env python3

import importlib.util
import json
import threading
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, Float64MultiArray, Int32, String


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class CTAG2F90DGripperNode(Node):

    def __init__(self):
        super().__init__("ctag2f90d_gripper_node")

        self.declare_parameter(
            "controller_module_path",
            "/home/kan/ros2_wss/gripper_modbus_final.py",
        )
        self.declare_parameter("arm_ip", "192.168.1.18")
        self.declare_parameter("arm_port", 8080)
        self.declare_parameter("device_id", 1)
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("switch_to_modbus_on_start", True)
        self.declare_parameter("enable_on_start", True)
        self.declare_parameter("reset_alarm_on_start", True)
        self.declare_parameter("follow_joint_move_to", True)
        self.declare_parameter("joint_move_to_topic", "/joint_move_to")
        self.declare_parameter("open_position", 0)
        self.declare_parameter("close_position", 9000)
        self.declare_parameter("position_max", 9000)
        self.declare_parameter("closed_input_value", 0.85)
        self.declare_parameter("default_speed", 50)
        self.declare_parameter("default_force", 30)
        self.declare_parameter("release_force", 100)
        self.declare_parameter("command_timeout", 5.0)
        self.declare_parameter("min_command_interval", 0.25)
        self.declare_parameter("position_deadband", 15)

        self._module_path = Path(str(self.get_parameter("controller_module_path").value))
        self._arm_ip = str(self.get_parameter("arm_ip").value)
        self._arm_port = int(self.get_parameter("arm_port").value)
        self._device_id = int(self.get_parameter("device_id").value)
        self._baudrate = int(self.get_parameter("baudrate").value)
        self._follow_joint_move_to = _as_bool(self.get_parameter("follow_joint_move_to").value)
        self._joint_move_to_topic = str(self.get_parameter("joint_move_to_topic").value)
        self._open_position = int(self.get_parameter("open_position").value)
        self._close_position = int(self.get_parameter("close_position").value)
        self._position_max = int(self.get_parameter("position_max").value)
        self._closed_input_value = float(self.get_parameter("closed_input_value").value)
        self._default_speed = int(self.get_parameter("default_speed").value)
        self._default_force = int(self.get_parameter("default_force").value)
        self._release_force = int(self.get_parameter("release_force").value)
        self._command_timeout = float(self.get_parameter("command_timeout").value)
        self._min_command_interval = float(self.get_parameter("min_command_interval").value)
        self._position_deadband = int(self.get_parameter("position_deadband").value)

        self._lock = threading.Lock()
        self._last_target_position = None
        self._last_command_time = 0.0
        self._last_result = {}

        self.status_pub = self.create_publisher(String, "/ctag2f90d/status", 10)
        self.grip_success_pub = self.create_publisher(Bool, "/ctag2f90d/grip_success", 10)

        self.create_subscription(Empty, "/ctag2f90d/open", self._open_cb, 10)
        self.create_subscription(Empty, "/ctag2f90d/close", self._close_cb, 10)
        self.create_subscription(Int32, "/ctag2f90d/target_position", self._target_position_cb, 10)
        self.create_subscription(String, "/ctag2f90d/command", self._command_cb, 10)

        if self._follow_joint_move_to:
            self.create_subscription(
                Float64MultiArray,
                self._joint_move_to_topic,
                self._joint_move_to_cb,
                10,
            )

        self._mod = self._load_controller_module(self._module_path)
        self._arm = self._create_arm()
        self._gripper = self._mod.GripperModbusController(
            self._arm,
            device_id=self._device_id,
            baudrate=self._baudrate,
        )

        if _as_bool(self.get_parameter("switch_to_modbus_on_start").value):
            if not self._gripper.switch_to_modbus_rtu_mode():
                raise RuntimeError("切换工具端 RS485 到 Modbus RTU 主站失败")

        if _as_bool(self.get_parameter("enable_on_start").value):
            ret = self._gripper.enable()
            if ret != 0:
                raise RuntimeError(f"夹爪电机使能失败: {ret}")

        if _as_bool(self.get_parameter("reset_alarm_on_start").value):
            self._gripper.reset_alarm()

        self.create_timer(0.2, self._publish_status)
        self.get_logger().info(
            "CTAG2F90D 夹爪节点已启动"
        )

    def _load_controller_module(self, module_path: Path):
        if not module_path.exists():
            raise FileNotFoundError(f"找不到夹爪控制文件: {module_path}")

        spec = importlib.util.spec_from_file_location("gripper_modbus_final", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载夹爪控制文件: {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _create_arm(self):
        arm = self._mod.RoboticArm(self._mod.rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = arm.rm_create_robot_arm(self._arm_ip, self._arm_port)
        handle_id = getattr(handle, "id", -1)
        if handle_id == -1:
            raise RuntimeError(f"连接机械臂失败: {self._arm_ip}:{self._arm_port}")
        self.get_logger().info(f"已连接机械臂: {self._arm_ip}:{self._arm_port}, handle={handle_id}")
        return arm

    def _open_cb(self, _msg: Empty):
        self._start_command_thread("open", self._open_position)

    def _close_cb(self, _msg: Empty):
        self._start_command_thread("close", self._close_position)

    def _target_position_cb(self, msg: Int32):
        self._start_command_thread("position", int(msg.data))

    def _command_cb(self, msg: String):
        command = msg.data.strip().lower()
        if command in ("open", "release"):
            self._start_command_thread("open", self._open_position)
        elif command in ("close", "grip"):
            self._start_command_thread("close", self._close_position)
        else:
            self.get_logger().warn(f"未知夹爪命令: {msg.data}")

    def _joint_move_to_cb(self, msg: Float64MultiArray):
        if len(msg.data) < 9:
            return
        gripper_value = float(msg.data[7])
        position = self._map_joint_move_gripper_value(gripper_value)
        self._start_command_thread("joint_move_to", position)

    def _map_joint_move_gripper_value(self, value: float) -> int:
        denom = self._closed_input_value if abs(self._closed_input_value) > 1e-6 else 1.0
        ratio = max(0.0, min(1.0, value / denom))
        pos = self._open_position + ratio * (self._close_position - self._open_position)
        return int(round(pos))

    def _start_command_thread(self, source: str, target_position: int):
        target_position = max(0, min(self._position_max, int(target_position)))
        now = time.time()
        if (
            self._last_target_position is not None
            and abs(target_position - self._last_target_position) <= self._position_deadband
            and now - self._last_command_time < self._min_command_interval
        ):
            return

        self._last_target_position = target_position
        self._last_command_time = now
        thread = threading.Thread(
            target=self._run_position_command,
            args=(source, target_position),
            daemon=True,
        )
        thread.start()

    def _run_position_command(self, source: str, target_position: int):
        if not self._lock.acquire(blocking=False):
            self.get_logger().warn("上一条夹爪命令仍在执行，忽略本次命令")
            return
        try:
            if target_position <= self._open_position + self._position_deadband:
                force = self._release_force
            else:
                force = self._default_force

            ret = self._gripper.move(
                position=target_position,
                speed=self._default_speed,
                force=force,
            )
            if ret != 0:
                result = {
                    "completed": False,
                    "grip_success": False,
                    "send_error": ret,
                }
            else:
                result = self._gripper.wait_for_motion_complete(
                    timeout=self._command_timeout,
                    verbose=False,
                )

            result["source"] = source
            result["target_position"] = target_position
            self._last_result = result
            self.grip_success_pub.publish(Bool(data=bool(result.get("grip_success", False))))
            self.get_logger().info(f"夹爪命令完成: {result}")
        except Exception as exc:
            self._last_result = {
                "source": source,
                "target_position": target_position,
                "error": str(exc),
            }
            self.get_logger().error(f"夹爪命令失败: {exc}")
        finally:
            self._lock.release()

    def _publish_status(self):
        try:
            status = self._gripper.get_full_status()
        except Exception as exc:
            status = {"error": str(exc)}
        status["last_result"] = self._last_result
        self.status_pub.publish(String(data=json.dumps(status, ensure_ascii=False)))

    def destroy_node(self):
        try:
            self._arm.rm_delete_robot_arm()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CTAG2F90DGripperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
