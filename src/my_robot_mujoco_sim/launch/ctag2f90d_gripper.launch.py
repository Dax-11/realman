from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'controller_module_path',
            default_value='/home/kan/ros2_wss/gripper_modbus_final.py',
        ),
        DeclareLaunchArgument('arm_ip', default_value='192.168.1.18'),
        DeclareLaunchArgument('arm_port', default_value='8080'),
        DeclareLaunchArgument('device_id', default_value='1'),
        DeclareLaunchArgument('baudrate', default_value='115200'),
        DeclareLaunchArgument('follow_joint_move_to', default_value='true'),
        DeclareLaunchArgument('open_position', default_value='0'),
        DeclareLaunchArgument('close_position', default_value='9000'),
        DeclareLaunchArgument('position_max', default_value='9000'),
        DeclareLaunchArgument('closed_input_value', default_value='0.85'),
        DeclareLaunchArgument('default_speed', default_value='50'),
        DeclareLaunchArgument('default_force', default_value='30'),
        DeclareLaunchArgument('command_timeout', default_value='5.0'),

        Node(
            package='my_robot_mujoco_sim',
            executable='ctag2f90d_gripper_node',
            name='ctag2f90d_gripper_node',
            output='screen',
            parameters=[{
                'controller_module_path': LaunchConfiguration('controller_module_path'),
                'arm_ip': LaunchConfiguration('arm_ip'),
                'arm_port': LaunchConfiguration('arm_port'),
                'device_id': LaunchConfiguration('device_id'),
                'baudrate': LaunchConfiguration('baudrate'),
                'follow_joint_move_to': LaunchConfiguration('follow_joint_move_to'),
                'open_position': LaunchConfiguration('open_position'),
                'close_position': LaunchConfiguration('close_position'),
                'position_max': LaunchConfiguration('position_max'),
                'closed_input_value': LaunchConfiguration('closed_input_value'),
                'default_speed': LaunchConfiguration('default_speed'),
                'default_force': LaunchConfiguration('default_force'),
                'command_timeout': LaunchConfiguration('command_timeout'),
            }],
        ),
    ])
