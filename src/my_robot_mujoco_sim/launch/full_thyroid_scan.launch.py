import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('my_robot_mujoco_sim')
    realsense_share = get_package_share_directory('realsense2_camera')

    enable_real_arg = DeclareLaunchArgument(
        'enable_real', default_value='true', description='是否尝试连接真实机械臂'
    )

    # 1. 机械臂与 MuJoCo 仿真驱动
    mujoco_real_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'mujoco_real.launch.py')
        ),
        launch_arguments={'enable_real': LaunchConfiguration('enable_real')}.items()
    )

    # 2. 真实 RealSense 相机驱动（独立加载，避免全局参数污染）
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(realsense_share, 'launch', 'rs_launch.py')
        ),
        launch_arguments={
            'align_depth.enable': 'true',
            'camera_name': 'camera',
            'enable_pointcloud': 'false'
        }.items()
    )

    # 3. 视觉定位与扫查节点
    thyroid_detector_node = Node(
        package='my_robot_mujoco_sim',
        executable='thyroid_detector_node',
        name='thyroid_3d_detector_node',
        output='screen'
    )

    thyroid_arm_control_node = Node(
        package='my_robot_mujoco_sim',
        executable='thyroid_arm_control_node',
        name='thyroid_arm_control_node',
        output='screen'
    )

    multi_view_scan_node = Node(
        package='my_robot_mujoco_sim',
        executable='multi_view_scan_node',
        name='multi_view_scan_node',
        output='screen'
    )

    return LaunchDescription([
        enable_real_arg,
        mujoco_real_launch,
        realsense_launch,
        thyroid_detector_node,
        thyroid_arm_control_node,
        multi_view_scan_node,
    ])