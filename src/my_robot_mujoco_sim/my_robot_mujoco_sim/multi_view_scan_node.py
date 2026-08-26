#!/usr/bin/env python3
import time
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException  # <--- 新增此行导入
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from ament_index_python.packages import get_package_share_directory
import mujoco
from my_robot_mujoco_sim.cartesian_controller import CartesianIKController


class MultiViewScanNode(Node):
    def __init__(self):
        super().__init__('multi_view_scan_node')
        
        pkg_share = get_package_share_directory('my_robot_mujoco_sim')
        scene_path = f"{pkg_share}/models/sim_scene.xml"
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        # 实时机械臂状态
        self.current_q_live = None
        self.current_joint_state_time = None
        
        self.ik_solver = CartesianIKController(self.model, site_name="gripper_tcp")
        
        self.move_pub = self.create_publisher(Float64MultiArray, '/joint_move_to', 10)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_cb, 10
        )
        self.pos_sub = self.create_subscription(
            PointStamped, '/thyroid_base_position', self._thyroid_pos_cb, 10
        )
        
        self.sphere_center = None
        self.radius = 0.15  # 扫查半径 15cm
        self.pre_move_duration = 3.0       # 预接近运动时间
        self.pre_dwell_time = 1.0          # 预接近停留时间

        self.scan_move_duration = 3.5      # 五个扫描视角之间的运动时间
        self.scan_dwell_time = 1.0         # 每个扫描视角停留时间
        
        self.viewpoints = [
            (-45, 0),
            (-20, 15),
            (0, 0),
            (20, 15),
            (45, 0)
        ]

    def _joint_state_cb(self, msg):
        joint_names = [f'joint{i}' for i in range(1, 8)]

        if all(name in msg.name for name in joint_names):
            q = np.zeros(7)

            for i, name in enumerate(joint_names):
                idx = msg.name.index(name)
                q[i] = msg.position[idx]

            self.current_q_live = q

        elif len(msg.position) >= 7:
            self.current_q_live = np.array(
                msg.position[:7],
                dtype=float
            )

        # ⭐ 每次收到 joint_states 都更新时间
        self.current_joint_state_time = time.time()

    def _thyroid_pos_cb(self, msg: PointStamped):
        if self.sphere_center is None:
            self.sphere_center = np.array([msg.point.x, msg.point.y, msg.point.z])
            self.get_logger().info(f"=== 成功接收锁定甲状腺坐标: X={msg.point.x:.3f}, Y={msg.point.y:.3f}, Z={msg.point.z:.3f} ===")

    def wait_until_arrived_and_hold(
        self,
        target_q,
        move_duration=7.0,
        tolerance=0.05,
        dwell_time=1.0,
        timeout_margin=5.0
    ):

        self.get_logger().info(
            "--> [WAIT START] 等待机械臂到达目标"
        )

        start_time = time.time()

        # 这里只限制“到达目标”的时间
        timeout = move_duration + timeout_margin

        last_log_time = 0.0

        # ==================================================
        # 第一阶段：等待到达目标
        # ==================================================
        while rclpy.ok():

            rclpy.spin_once(
                self,
                timeout_sec=0.05
            )

            elapsed = time.time() - start_time

            # ---------------- 超时判断 ----------------
            if elapsed > timeout:

                self.get_logger().error(
                    f"--> [TIMEOUT] "
                    f"{elapsed:.2f}s 仍未到达目标"
                )

                return False

            # ---------------- 没有 joint state ----------------
            if self.current_q_live is None:
                continue

            # ---------------- joint state 时间检查 ----------------
            if self.current_joint_state_time is None:
                continue

            state_age = (
                time.time()
                - self.current_joint_state_time
            )

            if state_age > 0.5:

                self.get_logger().warn(
                    f"--> joint_states 数据过期 "
                    f"{state_age:.3f}s"
                )

                continue

            # ---------------- 计算误差 ----------------
            joint_error = np.abs(
                self.current_q_live - target_q
            )

            max_error = np.max(
                joint_error
            )

            # ---------------- 定期打印 ----------------
            if (
                time.time()
                - last_log_time
                > 0.5
            ):

                last_log_time = time.time()

                self.get_logger().info(
                    f"--> [WAIT] "
                    f"t={elapsed:.2f}s | "
                    f"max_error={max_error:.4f} rad "
                    f"({np.degrees(max_error):.2f}°)"
                )

            # ==================================================
            # 到达目标
            # ==================================================
            if max_error <= tolerance:

                self.get_logger().info(
                    f"--> [ARRIVED] "
                    f"开始稳定保持，"
                    f"error={max_error:.4f} rad"
                )

                break

        # 如果 ROS 被关闭
        if not rclpy.ok():
            return False

        # ==================================================
        # 第二阶段：独立稳定保持
        # 注意：这里不再使用前面的运动 timeout
        # ==================================================

        hold_start = time.time()

        while rclpy.ok():

            rclpy.spin_once(
                self,
                timeout_sec=0.05
            )

            hold_elapsed = (
                time.time()
                - hold_start
            )

            # 保持时间完成
            if hold_elapsed >= dwell_time:

                self.get_logger().info(
                    f"--> [HOLD FINISHED] "
                    f"稳定保持 {hold_elapsed:.2f}s"
                )

                return True

            # 如果没有实时关节数据
            if self.current_q_live is None:
                continue

            # 继续检查是否偏离目标
            joint_error = np.abs(
                self.current_q_live
                - target_q
            )

            max_error = np.max(
                joint_error
            )

            # 如果保持期间偏离太多，可以重新计时
            if max_error > tolerance:

                self.get_logger().warn(
                    f"--> [HOLD UNSTABLE] "
                    f"保持期间误差变大: "
                    f"{max_error:.4f} rad，"
                    f"重新稳定"
                )

                hold_start = time.time()

        return False

    
    def compute_pose(self, az_deg, el_deg):
        az = np.radians(az_deg)
        el = np.radians(el_deg)
        
        dx = -self.radius * np.cos(el) * np.cos(az)
        dy = -self.radius * np.cos(el) * np.sin(az)
        dz = self.radius * np.sin(el)
        ee_pos = self.sphere_center + np.array([dx, dy, dz])
        
        z_axis = (self.sphere_center - ee_pos)
        z_axis /= np.linalg.norm(z_axis)
        
        up_world = np.array([0.0, 0.0, -1.0])
        x_axis = np.cross(up_world, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        
        R = np.column_stack((x_axis, y_axis, z_axis))
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R.flatten())
        return ee_pos, quat

    def run_scan(self):

        self.get_logger().info(
            "=== 等待 /joint_states 及 "
            "/thyroid_base_position 话题数据... ==="
        )

        while (
            self.current_q_live is None
            or self.sphere_center is None
        ) and rclpy.ok():

            rclpy.spin_once(
                self,
                timeout_sec=0.1
            )

        if not rclpy.ok():
            return

        self.get_logger().info(
            "=== 触发自动扫查任务：首先执行预接近过渡 ==="
        )

        current_q = self.current_q_live.copy()

        # ====================================================
        # 步骤 0：预接近
        # ====================================================

        pre_pos = self.sphere_center.copy()

        pre_pos[2] += 0.08

        pre_quat = np.array([
            0.0,
            0.0,
            1.0,
            0.0
        ])

        self.get_logger().info(
            f"--> [预接近阶段] "
            f"移动至正上方 8cm 处: "
            f"{np.round(pre_pos, 3)}"
        )

        target_q_pre = self.ik_solver.solve_ik(
            self.data,
            target_pos=pre_pos,
            target_quat=pre_quat,
            q_current=current_q,
            max_steps=200
        )

        for j in range(7):

            diff = (
                target_q_pre[j]
                - current_q[j]
            )

            target_q_pre[j] = (
                current_q[j]
                + np.arctan2(
                    np.sin(diff),
                    np.cos(diff)
                )
            )

        msg_pre = Float64MultiArray()

        msg_pre.data = (
            target_q_pre.tolist()
            + [self.pre_move_duration]
        )

        self.get_logger().info(
            "--> [PRE MOVE] 发布预接近目标"
        )

        self.move_pub.publish(
            msg_pre
        )

        # ====================================================
        # ⭐ 等待预接近真正完成
        # ====================================================

        success = self.wait_until_arrived_and_hold(
            target_q_pre,
            move_duration=self.pre_move_duration,
            dwell_time=self.pre_dwell_time
        )

        if not success:

            self.get_logger().error(
                "❌ 预接近失败，停止扫描任务"
            )

            return

        # 预接近成功后更新 current_q
        current_q = target_q_pre.copy()

        self.get_logger().info(
            "=== 预接近完成，开始执行 5 视角环绕扫描 ==="
        )

        # ====================================================
        # 步骤 1：5 个扫描视角
        # ====================================================

        for idx, (az, el) in enumerate(
            self.viewpoints
        ):

            # 计算目标位姿
            target_pos, target_quat = (
                self.compute_pose(
                    az,
                    el
                )
            )

            # 计算 IK
            target_q = self.ik_solver.solve_ik(
                self.data,
                target_pos=target_pos,
                target_quat=target_quat,
                q_current=current_q,
                max_steps=200
            )

            # 关节角连续化
            for j in range(7):

                diff = (
                    target_q[j]
                    - current_q[j]
                )

                target_q[j] = (
                    current_q[j]
                    + np.arctan2(
                        np.sin(diff),
                        np.cos(diff)
                    )
                )

            # ====================================================
            # 发布当前视角目标
            # ====================================================

            msg = Float64MultiArray()

            msg.data = (
                target_q.tolist()
                + [
                    self.scan_move_duration
                ]
            )

            self.get_logger().info(
                f"--> [视角 {idx + 1}/5] "
                f"发布目标 | "
                f"Az={az}°, "
                f"El={el}° | "
                f"目标位置="
                f"{np.round(target_pos, 3)}"
            )

            self.move_pub.publish(
                msg
            )

            # ====================================================
            # ⭐ 必须等待当前目标真正到达
            # ====================================================

            success = (
                self.wait_until_arrived_and_hold(
                    target_q,
                    move_duration=self.scan_move_duration,
                    dwell_time=self.scan_dwell_time
                )
            )

            # ====================================================
            # ⭐ 没成功，绝对不发送下一个目标
            # ====================================================

            if not success:

                self.get_logger().error(
                    f"❌ 视角 {idx + 1}/5 "
                    f"未成功到达，停止扫描"
                )

                return

            self.get_logger().info(
                f"✅ 视角 {idx + 1}/5 "
                f"完成"
            )

            # ====================================================
            # ⭐ 当前目标成功后，才更新 current_q
            # ====================================================

            current_q = target_q.copy()

        self.get_logger().info(
            "=== 环绕扫描任务顺利完成！"
            "机械臂保持在最后一个视角 ==="
        )


def main(args=None):
    rclpy.init(args=args)
    node = MultiViewScanNode()
    try:
        node.run_scan()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
