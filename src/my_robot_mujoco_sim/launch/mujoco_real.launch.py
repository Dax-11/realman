from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
# 把命令行传入的参数拿到 Node 里Node：
from launch.substitutions import LaunchConfiguration
# 启动某个 ROS2 节点
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('scene_file',     default_value='sim_scene.xml'),
        DeclareLaunchArgument('camera_name',    default_value='d435_view'),
        DeclareLaunchArgument('headless',       default_value='false'),
        DeclareLaunchArgument('move_duration',  default_value='3.0'),

        DeclareLaunchArgument('enable_real',    default_value='true',
                              description='是否尝试连接真臂'),
        DeclareLaunchArgument('real_ip',        default_value='192.168.1.18'),
        DeclareLaunchArgument('real_port',      default_value='8080'),
        DeclareLaunchArgument('stream_rate_hz', default_value='50.0'),
        DeclareLaunchArgument('follow_high',    default_value='false'),
        DeclareLaunchArgument('trajectory_mode', default_value='1'),
        DeclareLaunchArgument('radio',          default_value='50'),
        DeclareLaunchArgument('min_move_duration', default_value='2.0'),
        DeclareLaunchArgument('motion_start_hold_time', default_value='0.2'),
        DeclareLaunchArgument('send_idle_hold_position', default_value='true'),
        DeclareLaunchArgument('publish_real_state', default_value='true'),
        DeclareLaunchArgument('enable_gripper', default_value='false',
                              description='是否启动真实 CTAG2F90D 夹爪节点'),
        DeclareLaunchArgument(
            'gripper_module_path',
            default_value='/home/kan/ros2_wss/gripper_modbus_final.py'),
        DeclareLaunchArgument('gripper_device_id', default_value='1'),
        DeclareLaunchArgument('gripper_baudrate', default_value='115200'),
        DeclareLaunchArgument('gripper_open_position', default_value='0'),
        DeclareLaunchArgument('gripper_close_position', default_value='9000'),
        DeclareLaunchArgument('gripper_position_max', default_value='9000'),
        DeclareLaunchArgument('gripper_default_speed', default_value='50'),
        DeclareLaunchArgument('gripper_default_force', default_value='30'),

        Node(
            package='my_robot_mujoco_sim',
            executable='mujoco_real_node',
            name='mujoco_real_bridge_node',
            output='screen',
            parameters=[{
                'scene_file':     LaunchConfiguration('scene_file'),
                'camera_name':    LaunchConfiguration('camera_name'),
                'headless':       LaunchConfiguration('headless'),
                'move_duration':  LaunchConfiguration('move_duration'),

                'enable_real':    LaunchConfiguration('enable_real'),
                'real_ip':        LaunchConfiguration('real_ip'),
                'real_port':      LaunchConfiguration('real_port'),
                'stream_rate_hz': LaunchConfiguration('stream_rate_hz'),
                'follow_high':    LaunchConfiguration('follow_high'),
                'trajectory_mode': LaunchConfiguration('trajectory_mode'),
                'radio':          LaunchConfiguration('radio'),
                'min_move_duration': LaunchConfiguration('min_move_duration'),
                'motion_start_hold_time': LaunchConfiguration('motion_start_hold_time'),
                'send_idle_hold_position': LaunchConfiguration('send_idle_hold_position'),
                'publish_real_state': LaunchConfiguration('publish_real_state'),
                'enable_gripper': LaunchConfiguration('enable_gripper'),
                'gripper_module_path': LaunchConfiguration('gripper_module_path'),
                'gripper_device_id': LaunchConfiguration('gripper_device_id'),
                'gripper_baudrate': LaunchConfiguration('gripper_baudrate'),
                'gripper_open_position': LaunchConfiguration('gripper_open_position'),
                'gripper_close_position': LaunchConfiguration('gripper_close_position'),
                'gripper_position_max': LaunchConfiguration('gripper_position_max'),
                'gripper_default_speed': LaunchConfiguration('gripper_default_speed'),
                'gripper_default_force': LaunchConfiguration('gripper_default_force'),
            }],
        ),
    ])
