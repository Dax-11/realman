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
        
        self.ik_solver = CartesianIKController(self.model, site_name="gripper_tcp")
        
        self.move_pub = self.create_publisher(Float64MultiArray, '/joint_move_to', 10)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_cb, 10
        )
        self.pos_sub = self.create_subscription(
            PointStamped, '/thyroid_base_position', self._thyroid_pos_cb, 10
        )
        
        self.current_q_live = None
        self.sphere_center = None
        self.radius = 0.15  # 扫查半径 15cm
        
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
            self.current_q_live = np.array(msg.position[:7])

    def _thyroid_pos_cb(self, msg: PointStamped):
        if self.sphere_center is None:
            self.sphere_center = np.array([msg.point.x, msg.point.y, msg.point.z])
            self.get_logger().info(f"=== 成功接收锁定甲状腺坐标: X={msg.point.x:.3f}, Y={msg.point.y:.3f}, Z={msg.point.z:.3f} ===")

    def wait_until_arrived_and_hold(self, target_q, move_duration=7.0, tolerance=0.08, dwell_time=2.0):
        self.get_logger().info("--> 正在执行轨迹运动中...")
        time.sleep(max(1.0, move_duration - 2.0))
        
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            
            if self.current_q_live is not None:
                max_error = np.max(np.abs(self.current_q_live - target_q))
                if max_error <= tolerance:
                    self.get_logger().info(
                        f"--> [机械臂已到达目标] 关节残差: {np.round(max_error, 4)} rad ({np.round(np.degrees(max_error), 2)}°)"
                    )
                    self.get_logger().info(f"--> [开始静止停留 {dwell_time} 秒]...")
                    time.sleep(dwell_time)
                    self.get_logger().info("--> [停留结束] 准备触发下一个动作/视角\n")
                    return True
            time.sleep(0.1)

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
        self.get_logger().info("=== 等待 /joint_states 及 /thyroid_base_position 话题数据... ===")
        while (self.current_q_live is None or self.sphere_center is None) and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            
        self.get_logger().info("=== 触发自动扫查任务：首先执行预接近过渡 ===")
        current_q = self.current_q_live.copy()

        # ---------------- 步骤 0：预接近过渡 ----------------
        # 平移至甲状腺正上方 (Z 轴退让 8cm)，末端与 XY 平面平行 (quat=[0, 0, 1, 0])
        pre_pos = self.sphere_center.copy()
        pre_pos[2] += 0.08  # Z 轴退让 8cm
        pre_quat = np.array([0.0, 0.0, 1.0, 0.0])

        self.get_logger().info(f"--> [预接近阶段] 移动至正上方 8cm 处: {np.round(pre_pos, 3)}")
        target_q_pre = self.ik_solver.solve_ik(
            self.data,
            target_pos=pre_pos,
            target_quat=pre_quat,
            q_current=current_q,
            max_steps=200
        )
        for j in range(7):
            diff = target_q_pre[j] - current_q[j]
            target_q_pre[j] = current_q[j] + np.arctan2(np.sin(diff), np.cos(diff))

        msg_pre = Float64MultiArray()
        msg_pre.data = target_q_pre.tolist() + [5.0]  # 5秒移动
        self.move_pub.publish(msg_pre)
        
        self.wait_until_arrived_and_hold(target_q_pre, move_duration=5.0, dwell_time=1.5)
        current_q = target_q_pre.copy()

        # ---------------- 步骤 1：5 视角环绕扫描 ----------------
        self.get_logger().info("=== 开始执行 5 视角环绕扫描 ===")
        for idx, (az, el) in enumerate(self.viewpoints):
            target_pos, target_quat = self.compute_pose(az, el)
            
            target_q = self.ik_solver.solve_ik(
                self.data, 
                target_pos=target_pos, 
                target_quat=target_quat,
                q_current=current_q,
                max_steps=200
            )
            
            for j in range(7):
                diff = target_q[j] - current_q[j]
                target_q[j] = current_q[j] + np.arctan2(np.sin(diff), np.cos(diff))
                
            current_q = target_q.copy()
            
            msg = Float64MultiArray()
            msg.data = target_q.tolist() + [7.0]
            
            self.get_logger().info(f"--> [视角 {idx+1}/5] 发布目标: Az={az}°, El={el}° | 目标位置: {np.round(target_pos, 3)}")
            self.move_pub.publish(msg)
            
            self.wait_until_arrived_and_hold(target_q, move_duration=7.0, dwell_time=2.0)
            
        self.get_logger().info("=== 环绕扫描任务顺利完成！机械臂保持在最后一个视角停顿 ===")


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