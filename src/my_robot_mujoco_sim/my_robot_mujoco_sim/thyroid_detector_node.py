#!/usr/bin/env python3
import time
import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException  # <--- 新增此行导入
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge, CvBridgeError
import mediapipe as mp


class Thyroid3DDetectorNode(Node):
    def __init__(self):
        super().__init__('thyroid_3d_detector_node')
        
        self.bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None
        
        # 1. 增加默认内参兜底（防止缺失 CameraInfo 时整个节点挂起）
        self.intrinsics = {
            'fx': 615.0, 'fy': 615.0,
            'cx': 320.0, 'cy': 240.0,
            'frame_id': 'camera_color_optical_frame'
        }
        
        # 时域滤波缓存 (Exponential Moving Average)
        self.smooth_3d = None
        self.alpha = 0.3  # 平滑系数 (0~1)

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 绑定真实 RealSense D435 相机的话题
        self.color_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self._color_cb, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self._depth_cb, 10
        )
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self._info_cb, 10
        )
        
        self.point_pub = self.create_publisher(PointStamped, '/adam_apple_3d_position', 10)
        self.timer = self.create_timer(0.033, self.process_frame)

    def _info_cb(self, msg):
        K = msg.k
        self.intrinsics = {
            'fx': K[0], 'cx': K[2],
            'fy': K[4], 'cy': K[5],
            'frame_id': msg.header.frame_id
        }

    def _color_cb(self, msg):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError:
            pass

    def _depth_cb(self, msg):
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_depth = depth_img.astype(np.float32) / 1000.0 if depth_img.dtype == np.uint16 else depth_img
        except CvBridgeError:
            pass

    def detect_thyroid_roi_enhanced(self, color_img, face_lm, pose_lm, w, h):
        """基于多特征点几何约束与姿态向量推算喉结/甲状腺 2D 位置"""
        nose = np.array([face_lm[1].x * w, face_lm[1].y * h])
        chin = np.array([face_lm[152].x * w, face_lm[152].y * h])
        jaw_l = np.array([face_lm[172].x * w, face_lm[172].y * h])
        jaw_r = np.array([face_lm[397].x * w, face_lm[397].y * h])
        
        ls = np.array([pose_lm[11].x * w, pose_lm[11].y * h])
        rs = np.array([pose_lm[12].x * w, pose_lm[12].y * h])
        shoulder_center = (ls + rs) / 2.0

        # 1. 计算头部中轴延伸向量 (鼻尖 -> 下巴)
        head_vec = chin - nose
        norm = np.linalg.norm(head_vec)
        if norm < 1e-3:
            return int(chin[0]), int(chin[1] + 30)
        head_dir = head_vec / norm

        # 2. 估计颈部长度（下巴到双肩中点的距离）
        neck_len = np.linalg.norm(shoulder_center - chin)
        
        # 解剖学上喉结通常位于下巴下方约 35% ~ 45% 颈长位置
        estimated_apple = chin + head_dir * (neck_len * 0.40)

        # 3. 横向宽度约束（基于下颚角距离）
        jaw_width = np.linalg.norm(jaw_r - jaw_l)
        search_radius = int(jaw_width * 0.3)

        u_center, v_center = int(estimated_apple[0]), int(estimated_apple[1])
        
        # 在局部搜索框内结合梯度微调
        roi_x1 = max(0, u_center - search_radius)
        roi_x2 = min(w - 1, u_center + search_radius)
        roi_y1 = max(0, v_center - search_radius)
        roi_y2 = min(h - 1, v_center + search_radius)

        if roi_x2 > roi_x1 and roi_y2 > roi_y1:
            roi = color_img[roi_y1:roi_y2, roi_x1:roi_x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.abs(laplacian)
            
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(laplacian)
            if max_val > 20:
                u_center = roi_x1 + max_loc[0]
                v_center = roi_y1 + max_loc[1]

        # 绘制特征标注
        cv2.polylines(color_img, [np.int32([jaw_l, chin, jaw_r])], False, (255, 255, 0), 2)
        cv2.line(color_img, (int(chin[0]), int(chin[1])), (int(shoulder_center[0]), int(shoulder_center[1])), (0, 255, 255), 1)
        cv2.rectangle(color_img, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 1)

        return u_center, v_center

    def process_frame(self):
        if self.latest_color is None:
            self.get_logger().warn(
                "等待彩色图像数据中 (/camera/camera/color/image_raw)...",
                throttle_duration_sec=3.0
            )
            return

        # ==================== 图像方向纠正配置 ====================
        #  -1 : 旋转 180 度 (上下+左右同时翻转，常用于相机倒挂安装)
        #   1 : 左右镜像翻转 (左右颠倒时使用)
        #   0 : 上下镜像翻转 (上下颠倒时使用)
        FLIP_MODE = 0  

        color_img = cv2.flip(self.latest_color.copy(), FLIP_MODE)

        if self.latest_depth is None:
            cv2.putText(color_img, "Waiting for Depth Image...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Thyroid 3D Detection", color_img)
            cv2.waitKey(1)
            return

        # 深度图与彩色图同步翻转，确保像素 1:1 对齐
        depth_img = cv2.flip(self.latest_depth.copy(), FLIP_MODE)
        h, w, _ = color_img.shape
        # =========================================================

        rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_img)

        # 未识别到人物特征时，显示搜索状态画框
        if not results.pose_landmarks or not results.face_landmarks:
            cv2.putText(color_img, "Searching for Target...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Thyroid 3D Detection", color_img)
            cv2.waitKey(1)
            return

        # 增强版 2D 提取
        u_apple, v_apple = self.detect_thyroid_roi_enhanced(
            color_img, results.face_landmarks.landmark, results.pose_landmarks.landmark, w, h
        )

        # 5x5 深度窗口中值滤波
        u_c = max(2, min(u_apple, w - 3))
        v_c = max(2, min(v_apple, h - 3))
        depth_patch = depth_img[v_c-2:v_c+3, u_c-2:u_c+3]
        valid_depths = depth_patch[(depth_patch > 0.2) & (depth_patch < 2.0)]

        if len(valid_depths) > 0:
            z_raw = float(np.median(valid_depths))
            fx, fy = self.intrinsics['fx'], self.intrinsics['fy']
            cx, cy = self.intrinsics['cx'], self.intrinsics['cy']
            
            x_raw = (u_apple - cx) * z_raw / fx
            y_raw = (v_apple - cy) * z_raw / fy

            curr_3d = np.array([x_raw, y_raw, z_raw])

            # 时域平滑滤波 (EMA)
            if self.smooth_3d is None:
                self.smooth_3d = curr_3d
            else:
                self.smooth_3d = self.alpha * curr_3d + (1 - self.alpha) * self.smooth_3d

            x, y, z = self.smooth_3d

            # 发布 ROS2 消息
            pt_msg = PointStamped()
            pt_msg.header.stamp = self.get_clock().now().to_msg()
            pt_msg.header.frame_id = self.intrinsics['frame_id']
            pt_msg.point.x, pt_msg.point.y, pt_msg.point.z = float(x), float(y), float(z)
            self.point_pub.publish(pt_msg)

            # 可视化
            cv2.circle(color_img, (u_apple, v_apple), 6, (0, 0, 255), -1)
            cv2.putText(color_img, f"Thyroid 3D: X={x:.3f}m Y={y:.3f}m Z={z:.3f}m", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Thyroid 3D Detection", color_img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = Thyroid3DDetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        # 4. 安全优雅退出
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()