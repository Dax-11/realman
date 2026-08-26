#!/usr/bin/env python3
# 导入标准库：操作系统路径处理
import os
# 导入标准库：多线程模块，实现渲染/仿真/ROS通信的线程分离
import threading
# 导入数值计算库：用于关节位置插值、数组操作、图像数据处理
import numpy as np

# 导入ROS2核心库：Python版ROS2基础依赖
import rclpy
# 导入ROS2节点基类：所有自定义ROS2节点都需要继承此类
from rclpy.node import Node
# 导入ROS2工具函数：获取功能包的共享目录（用于加载模型文件）
from ament_index_python.packages import get_package_share_directory

# 导入MuJoCo核心库：物理仿真引擎
import mujoco
# 导入MuJoCo可视化模块：用于创建仿真窗口
import mujoco.viewer

# 导入OpenCV库：用于显示相机视角画面，兼容未安装的情况
try:
    # pyrefly: ignore [missing-import]
    import cv2
    # 标记：是否成功加载OpenCV
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# 导入ROS2标准消息：关节状态消息（用于发布/joint_states）
from sensor_msgs.msg import JointState
# 导入ROS2标准消息：浮点多维数组（用于接收关节控制指令）
from std_msgs.msg import Float64MultiArray
# 导入ROS2标准服务：触发型服务（用于复位仿真）
from std_srvs.srv import Trigger


# 自定义ROS2节点类：继承ROS2 Node基类，封装RM75机械臂MuJoCo仿真所有逻辑
class MuJoCoSimNode(Node):

    # 类常量：RM75机械臂的7个关节名称（必须与MJCF/URDF模型中的关节名完全一致）
    JOINT_NAMES = [
        'joint1', 'joint2', 'joint3',
        'joint4', 'joint5', 'joint6', 'joint7'
    ]

    # 构造函数：节点初始化入口，完成所有参数、模型、通信、线程的初始化
    def __init__(self):
        # 调用父类构造函数，初始化ROS2节点，节点名称：mujoco_sim_node
        super().__init__('mujoco_sim_node')

        # ===================== 1. 声明ROS2参数（可通过命令行/launch文件修改） =====================
        # 场景模型文件（MJCF格式.xml）
        self.declare_parameter('scene_file',     'sim_scene.xml')
        # 仿真中相机的名称
        self.declare_parameter('camera_name',    'd435_view')
        # 是否启用无头模式（True=无可视化窗口，仅后台仿真）
        self.declare_parameter('headless',       False)
        # 默认平滑运动时长（秒）
        self.declare_parameter('move_duration',  3.0)

        # 获取参数值：将ROS2参数赋值给本地变量
        scene_file  = self.get_parameter('scene_file').value
        self._cam   = self.get_parameter('camera_name').value
        headless    = self.get_parameter('headless').value
        self._traj_default_dur = self.get_parameter('move_duration').value

        # ===================== 2. 定位仿真模型文件路径（ROS2功能包标准路径） =====================
        # 获取当前功能包my_robot_mujoco_sim的共享目录（install目录下）
        pkg_share  = get_package_share_directory('my_robot_mujoco_sim')
        # 拼接场景文件完整路径：功能包目录/models/xxx.xml
        scene_path = os.path.join(pkg_share, 'models', scene_file)
        # 拼接机械臂网格模型路径：功能包目录/models/RM75_6F（存放STL网格文件）
        mesh_dir   = os.path.join(pkg_share, 'models', 'RM75_6F')

        # 日志打印：输出加载的场景文件路径
        self.get_logger().info(f'场景文件: {scene_path}')

        # ===================== 3. 加载模型网格资源（兼容MJCF/URDF两种模型格式） =====================
        # 调用自定义方法：加载STL网格文件到字典，MuJoCo加载模型时使用
        assets = self._load_assets(mesh_dir)

        # ===================== 4. 加载MuJoCo仿真模型 =====================
        try:
            # 如果有网格资源，带资源加载模型
            if assets:
                self.model = mujoco.MjModel.from_xml_path(
                    scene_path, assets=assets)
            # 无额外资源，直接加载模型
            else:
                self.model = mujoco.MjModel.from_xml_path(scene_path)
        # 捕获模型加载异常（文件不存在/格式错误/路径错误）
        except Exception as e:
            self.get_logger().error(f'模型加载失败: {e}')
            # 抛出异常，终止节点运行
            raise

        # 创建MuJoCo数据对象：存储仿真实时状态（关节位置、速度、控制力、碰撞信息等）
        self.data = mujoco.MjData(self.model)

        # ===================== 5. 复位机械臂到HOME关键帧（初始姿态） =====================
        self._apply_home_keyframe()

        # ===================== 6. 检查仿真相机是否存在 =====================
        # MuJoCo函数：通过名称获取相机ID，-1表示不存在
        cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, self._cam)
        # 标记：相机是否有效
        self._has_camera = (cam_id != -1)
        if not self._has_camera:
            self.get_logger().warn(
                f"相机 '{self._cam}' 不存在，OpenCV 窗口将显示提示")

        # ===================== 7. 多线程安全锁 =====================
        # 线程锁：防止仿真线程、渲染线程、ROS回调线程同时修改共享数据（线程竞争）
        self._lock = threading.Lock()

        # ===================== 8. 平滑轨迹运动状态变量 =====================
        # 轨迹起始关节位置
        self._traj_start_pos  = None
        # 轨迹目标关节位置
        self._traj_target_pos = None
        # 轨迹开始的仿真时间
        self._traj_start_time = None
        # 轨迹运动总时长
        self._traj_duration   = self._traj_default_dur

        # ===================== 9. ROS2 通信接口（发布/订阅/服务） =====================
        # 这里只是创建了，没有参数接着
        # 创建发布者：话题/joint_states，消息类型JointState，队列长度10
        self.joint_pub = self.create_publisher(
            JointState, '/joint_states', 10)

        # 创建订阅者：订阅/joint_commands，回调函数_cmd_callback，队列长度10
        self.create_subscription(
            Float64MultiArray, '/joint_commands',
            self._cmd_callback, 10)

        # 创建订阅者：订阅/joint_move_to，回调函数_move_to_callback，队列长度10
        self.create_subscription(
            Float64MultiArray, '/joint_move_to',
            self._move_to_callback, 10)

        # 创建服务：服务名/mujoco/reset，服务类型Trigger，回调函数_reset_callback
        self.create_service(
            Trigger, '/mujoco/reset', self._reset_callback)

        # ===================== 10. 仿真步进定时器 =====================
        # 获取MuJoCo模型的仿真时间步长（秒）
        dt = float(self.model.opt.timestep)
        # 创建定时器：每dt毫秒执行一次_sim_step，实现固定步长仿真步进
        self.create_timer(dt, self._sim_step)

        # ===================== 11. 启动可视化渲染线程 =====================
        # 非无头模式：启动可视化窗口
        if not headless:
            # 创建子线程：目标函数_render_loop，守护线程（主进程退出时自动退出）
            t = threading.Thread(
                target=self._render_loop,
                daemon=True,
                name='mujoco_render')
            # 启动渲染线程
            t.start()
        # 无头模式：仅后台仿真，不启动窗口
        else:
            self.get_logger().info('无头模式')

        # 日志打印：节点启动成功，输出核心参数
        self.get_logger().info(
            f'节点启动！timestep={dt*1000:.1f}ms  '
            f'nu={self.model.nu}  nq={self.model.nq}'
        )

    # ===================== 私有方法：加载网格模型资源 =====================
    def _load_assets(self, mesh_dir: str) -> dict:
        # 初始化空资源字典
        assets = {}
        # 目录不存在，直接返回空字典
        if not os.path.exists(mesh_dir):
            return assets

        # 遍历目录下所有文件
        for fname in os.listdir(mesh_dir):
            # 仅加载STL格式的网格文件（不区分大小写）
            if fname.lower().endswith('.stl'):
                # 拼接文件完整路径
                fpath = os.path.join(mesh_dir, fname)
                # 以二进制模式读取文件
                with open(fpath, 'rb') as f:
                    # 字典赋值：文件名作为key，二进制数据作为value
                    assets[fname] = f.read()

        # 打印加载的网格数量
        if assets:
            self.get_logger().info(f'加载了 {len(assets)} 个 STL 资源')
        return assets

    # ===================== 私有方法：应用HOME关键帧（复位初始姿态） =====================
    def _apply_home_keyframe(self):
        # 模型中无关键帧，直接返回
        if self.model.nkey <= 0:
            return

        # 通过名称获取home关键帧的ID
        home_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, 'home')

        # 未找到home关键帧，打印警告
        if home_id == -1:
            self.get_logger().warn("找不到 'home' 关键帧，使用默认姿态")
            return

        # 赋值：将关键帧的关节位置赋值给仿真数据
        self.data.qpos[:] = self.model.key_qpos[home_id]
        # 赋值：将关键帧的控制量赋值给仿真数据（有执行器时生效）
        if self.model.nu > 0:
            self.data.ctrl[:] = self.model.key_ctrl[home_id]
        # MuJoCo前向动力学计算：更新所有状态（位置/速度/姿态）
        mujoco.mj_forward(self.model, self.data)
        self.get_logger().info("已应用 home 关键帧")

    # ===================== 私有方法：仿真步进回调（定时器核心函数） =====================
    def _sim_step(self):
        # 加线程锁：保护共享数据（仿真状态）不被多线程同时修改
        with self._lock:
            # 更新轨迹插值（平滑运动）
            self._update_trajectory()
            # 执行一步MuJoCo物理仿真（核心：计算动力学、碰撞、运动）
            mujoco.mj_step(self.model, self.data)
        # 发布当前关节状态到ROS2话题
        self._publish_joint_states()

    # ===================== 私有方法：平滑轨迹插值计算 =====================
    def _update_trajectory(self):
        # 无目标轨迹，直接返回
        if self._traj_target_pos is None:
            return
        # 计算轨迹已运行时间（当前仿真时间 - 轨迹启动时间）
        elapsed = self.data.time - self._traj_start_time
        # 计算插值系数：0~1（超过1则固定为1）
        alpha   = min(elapsed / self._traj_duration, 1.0)
        # 五次Smoothstep公式：生成平滑的插值曲线（加速度连续，无抖动）
        s = alpha ** 3 * (alpha * (alpha * 6 - 15) + 10)
        # 插值计算：实时更新关节控制量（前7个值对应7个关节）
        self.data.ctrl[:7] = (
            self._traj_start_pos
            + s * (self._traj_target_pos - self._traj_start_pos)
        )
        # 插值完成：清空目标轨迹，标记运动结束
        if alpha >= 1.0:
            self._traj_target_pos = None
            self.get_logger().info('运动完成')

    # ===================== 私有方法：MuJoCo可视化+OpenCV相机渲染循环 =====================
    def _render_loop(self):
        # OpenCV初始化：仅在安装OpenCV时生效
        if HAS_CV2:
            # 创建MuJoCo渲染器：分辨率640x480
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            # 创建OpenCV窗口：名称=相机名，可缩放
            cv2.namedWindow(self._cam, cv2.WINDOW_NORMAL)
        else:
            renderer = None
            self.get_logger().warn('未检测到 OpenCV')

        # 启动MuJoCo被动式查看器：非阻塞，仅渲染不控制仿真
        with mujoco.viewer.launch_passive(
            self.model, self.data,
            show_left_ui=True,
            show_right_ui=True,
        ) as viewer:

            self.get_logger().info(' MuJoCo 窗口已打开')

            # 窗口运行循环：只要窗口未关闭就持续执行
            while viewer.is_running():

                # 1. 同步MuJoCo窗口状态（线程安全）
                with self._lock:
                    viewer.sync()

                # 2. OpenCV相机画面渲染
                if renderer is not None:
                    # 相机存在：渲染仿真相机画面
                    if self._has_camera:
                        with self._lock:
                            # 更新渲染场景：指定相机视角
                            renderer.update_scene(
                                self.data, camera=self._cam)
                        # 获取渲染后的RGB图像
                        image = renderer.render()
                        # 转换颜色空间：RGB→BGR（OpenCV默认格式），并显示
                        cv2.imshow(
                            self._cam,
                            cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    # 相机不存在：显示红色提示文字
                    else:
                        # 创建纯白背景图
                        blank = np.full((480, 640, 3), 255, dtype=np.uint8)
                        # 绘制提示文字
                        cv2.putText(
                            blank,
                            f"Camera '{self._cam}' not found",
                            (40, 240),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
                        cv2.imshow(self._cam, blank)

                    # 监听键盘按键：按Q退出OpenCV窗口
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break

                # 控制渲染帧率：约100Hz，降低CPU占用
                threading.Event().wait(timeout=0.01)

            # 窗口关闭后：资源清理
            if renderer is not None:
                renderer.close()
                cv2.destroyAllWindows()

        self.get_logger().info('MuJoCo 窗口已关闭')

    # ===================== ROS回调：/joint_commands 直接跳位控制 =====================
    def _cmd_callback(self, msg: Float64MultiArray):
        # 校验数据长度：必须是7个关节值
        if len(msg.data) != 7:
            self.get_logger().warn(f'期望7个值，收到{len(msg.data)}')
            return
        # 线程安全：直接设置控制量
        with self._lock:
            # 清空轨迹：停止平滑运动
            self._traj_target_pos = None
            # 直接赋值关节控制量
            self.data.ctrl[:7] = np.array(msg.data)

    def _move_to_callback(self, msg: Float64MultiArray):
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

        with self._lock:
            self._traj_start_pos  = self.data.qpos[:7].copy()
            self._traj_target_pos = target
            self._traj_start_time = self.data.time
            self._traj_duration   = duration
            if gripper_target is not None and self.model.nu > 7:
                self.data.ctrl[7] = gripper_target
        self.get_logger().info(
            f'→ 运动目标: {np.round(target,3).tolist()} | 夹爪: {gripper_target} | 时长={duration}s')

    # ===================== ROS服务回调：/mujoco/reset 仿真复位 =====================
    def _reset_callback(self, request, response):
        # 线程安全：复位操作
        with self._lock:
            # 清空轨迹
            self._traj_target_pos = None
            # 应用HOME关键帧
            self._apply_home_keyframe()
        # 填充响应数据
        response.success = True
        response.message = '已重置到 home'
        return response

    # ===================== 私有方法：发布/joint_states 关节状态 =====================
    def _publish_joint_states(self):
        # 创建消息对象
        msg = JointState()
        # 填充时间戳：当前ROS2时间
        msg.header.stamp = self.get_clock().now().to_msg()
        # 填充关节名称
        msg.name     = self.JOINT_NAMES
        # 填充关节位置（前7个关节）
        msg.position = self.data.qpos[:7].tolist()
        # 填充关节速度（前7个关节）
        msg.velocity = self.data.qvel[:7].tolist()
        # 填充关节力矩/执行器力（前7个关节）
        msg.effort   = self.data.actuator_force[:7].tolist()
        # 发布消息
        self.joint_pub.publish(msg)


# ===================== 主函数：节点入口 =====================
def main(args=None):
    # 初始化ROS2客户端库
    rclpy.init(args=args)
    # 创建自定义仿真节点实例
    node = MuJoCoSimNode()
    try:
        # 启动ROS2自旋：保持节点运行，处理回调/话题/服务
        rclpy.spin(node)
    # 捕获键盘中断（Ctrl+C）
    except KeyboardInterrupt:
        pass
    finally:
        # 资源清理：销毁节点
        node.destroy_node()
        # 关闭ROS2
        rclpy.shutdown()


# Python主入口：执行main函数
if __name__ == '__main__':
    main()
