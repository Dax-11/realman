#!/usr/bin/env python3
import os
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException  # <--- 新增此行导入
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from ament_index_python.packages import get_package_share_directory

import mujoco
from my_robot_mujoco_sim.cartesian_controller import CartesianIKController


class ThyroidArmControlBridgeNode(Node):
    def __init__(self):
        super().__init__('thyroid_arm_control_node')

        self.declare_parameter('scene_file', 'sim_scene.xml')
        self.declare_parameter('lock_stable_frames', 15)

        scene_file = self.get_parameter('scene_file').value
        self.lock_threshold = self.get_parameter('lock_stable_frames').value

        # 加载 MuJoCo 模型仅用于实时提取相机外参 T_base_cam
        pkg_share = get_package_share_directory('my_robot_mujoco_sim')
        scene_path = os.path.join(pkg_share, 'models', scene_file)
        mesh_dir = os.path.join(pkg_share, 'models', 'RM75_6F')

        assets = self._load_assets(mesh_dir)
        self.model = mujoco.MjModel.from_xml_path(scene_path, assets=assets) if assets else mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.ik_solver = CartesianIKController(self.model, site_name="gripper_tcp")

        self.current_qpos = np.zeros(7)
        self.target_history = []
        self.is_locked = False

        self.joint_sub = self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        self.vision_sub = self.create_subscription(PointStamped, '/adam_apple_3d_position', self._vision_cb, 10)
        self.base_pos_pub = self.create_publisher(PointStamped, '/thyroid_base_position', 10)

        self.get_logger().info("=== 甲状腺坐标转换节点已就绪，等待视觉 3D 识别 ===")

    def _load_assets(self, mesh_dir: str) -> dict:
        assets = {}
        if os.path.exists(mesh_dir):
            for fname in os.listdir(mesh_dir):
                if fname.lower().endswith('.stl'):
                    with open(os.path.join(mesh_dir, fname), 'rb') as f:
                        assets[fname] = f.read()
        return assets

    def _joint_cb(self, msg: JointState):
        if len(msg.position) >= 7:
            self.current_qpos = np.array(msg.position[:7])

    def _vision_cb(self, msg: PointStamped):
        if self.is_locked:
            return

        # 1. 提取 ROS 相机 Optical Frame 坐标 [X_right, Y_down, Z_depth]
        p_cam_ros = np.array([msg.point.x, msg.point.y, msg.point.z])

        # 2. 转换至 MuJoCo 相机坐标系 [X_right, Y_up, Z_back]
        p_cam_mj = np.array([p_cam_ros[0], -p_cam_ros[1], -p_cam_ros[2]])

        # 3. 实时同步机械臂位姿并提取相机位姿
        self.data.qpos[self.ik_solver.qpos_start : self.ik_solver.qpos_start + 7] = self.current_qpos
        mujoco.mj_forward(self.model, self.data)

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "d435_view")
        if cam_id < 0:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "table_cam")

        T_base_cam = np.eye(4)
        if cam_id >= 0:
            T_base_cam[:3, 3] = self.data.cam_xpos[cam_id]
            T_base_cam[:3, :3] = self.data.cam_xmat[cam_id].reshape(3, 3)

        # 4. 转换至机械臂基座坐标系
        p_base = T_base_cam[:3, :3] @ p_cam_mj + T_base_cam[:3, 3]

        self.get_logger().info(
            f"[坐标转换] 相机深度 Z={p_cam_ros[2]:.3f}m ==> 基座坐标: X={p_base[0]:.3f}, Y={p_base[1]:.3f}, Z={p_base[2]:.3f}m",
            throttle_duration_sec=1.0
        )

        self.target_history.append(p_base)

        if len(self.target_history) > self.lock_threshold:
            self.target_history.pop(0)

            std_dev = np.std(self.target_history, axis=0)
            if np.all(std_dev < 0.005):  # 波动 < 5mm 锁定
                locked_pos = np.mean(self.target_history, axis=0)
                self.is_locked = True
                self.get_logger().info(
                    f"★ 目标精准锁定! 基座坐标: X={locked_pos[0]:.3f}, Y={locked_pos[1]:.3f}, Z={locked_pos[2]:.3f}"
                )

                out_msg = PointStamped()
                out_msg.header.stamp = self.get_clock().now().to_msg()
                out_msg.header.frame_id = 'base_link'
                out_msg.point.x, out_msg.point.y, out_msg.point.z = float(locked_pos[0]), float(locked_pos[1]), float(locked_pos[2])
                self.base_pos_pub.publish(out_msg)
        # 在 p_base = (T_base_cam @ p_cam)[:3] 这一行之前，插入：
        self.get_logger().info(f"--- 调试信息 ---")
        self.get_logger().info(f"Camera T_base_cam (位姿):\n{T_base_cam}")
        self.get_logger().info(f"P_cam_mj (输入): {p_cam_mj}")
        # 确认基座位置是否为原点
        cam_pos_world = self.data.cam_xpos[cam_id]
        self.get_logger().info(f"Camera World Pos: {cam_pos_world}")


def main(args=None):
    rclpy.init(args=args)
    node = ThyroidArmControlBridgeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()