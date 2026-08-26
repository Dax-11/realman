from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        DeclareLaunchArgument(
            'scene_file',
            default_value='sim_scene.xml',
            description='models/ 目录下的场景文件名'
        ),
        DeclareLaunchArgument(
            'camera_name',
            default_value='d435_view',
            description='MJCF 里定义的相机名称'
        ),
        DeclareLaunchArgument(
            'headless',
            default_value='false',
            description='true = 不弹窗口'
        ),
        DeclareLaunchArgument(
            'move_duration',
            default_value='3.0',
            description='平滑运动默认时长（秒）'
        ),

        # 启动仿真节点
        Node(
            package='my_robot_mujoco_sim',
            executable='mujoco_node',
            name='mujoco_sim_node',
            output='screen',
            parameters=[{
                'scene_file':    LaunchConfiguration('scene_file'),
                'camera_name':   LaunchConfiguration('camera_name'),
                'headless':      LaunchConfiguration('headless'),
                'move_duration': LaunchConfiguration('move_duration'),
            }],
        ),
    ])