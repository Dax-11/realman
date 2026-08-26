from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('arm_variant', default_value='6f',
                              description='RM75 夹爪变体：6f 或 6fb'),
        DeclareLaunchArgument('scene_file', default_value='sim_scene.xml'),
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('enable_mujoco', default_value='true',
                              description='是否启动 MuJoCo 镜像窗口'),
        DeclareLaunchArgument('enable_gripper', default_value='false',
                              description='是否启动真实 CTAG2F90D 夹爪节点'),
        DeclareLaunchArgument(
            'gripper_module_path',
            default_value='/home/kan/ros2_wss/gripper_modbus_final.py'),
        DeclareLaunchArgument('real_ip', default_value='192.168.1.18'),
        DeclareLaunchArgument('real_port', default_value='8080'),
        DeclareLaunchArgument('gripper_device_id', default_value='1'),
        DeclareLaunchArgument('gripper_baudrate', default_value='115200'),
        DeclareLaunchArgument('gripper_default_speed', default_value='50'),
        DeclareLaunchArgument('gripper_default_force', default_value='30'),

        # 1) rm_driver：官方真机驱动（发 /joint_states、接 /rm_driver/* 指令）
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('rm_driver'),
                    'launch', 'rm_75_driver.launch.py'])]),
        ),

        # 1.5) rm_control：为 MoveIt 提供 FollowJointTrajectory action，
        #      再桥接到 rm_driver 的透传指令
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('rm_control'),
                    'launch', 'rm_75_control.launch.py'])]),
        ),

        # 2) MoveIt2：规划 + RViz
        TimerAction(
            period=2.0,  # 给 rm_driver 一点时间建立 /joint_states
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([
                        # 旧方案保留作参考：直接使用官方 6F MoveIt 描述。
                        # PathJoinSubstitution([
                        #     FindPackageShare('rm_75_config'),
                        #     'launch', 'real_moveit_demo_6f.launch.py'])]),
                        PathJoinSubstitution([
                            FindPackageShare('my_robot_mujoco_sim'),
                            'launch', 'real_moveit_demo_mujoco_6f.launch.py'])]),
                    condition=IfCondition(
                        PythonExpression(["'", LaunchConfiguration('arm_variant'), "' == '6f'"])
                    ),
                ),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([
                        PathJoinSubstitution([
                            FindPackageShare('rm_75_config'),
                            'launch', 'real_moveit_demo_6fb.launch.py'])]),
                    condition=IfCondition(
                        PythonExpression(["'", LaunchConfiguration('arm_variant'), "' == '6fb'"])
                    ),
                ),
            ],
        ),

        # 3) MuJoCo LISTEN：镜像可视化
        TimerAction(
            period=4.0,  # MoveIt 完全拉起后再开窗
            actions=[
                Node(
                    package='my_robot_mujoco_sim',
                    executable='mujoco_real_node',
                    name='mujoco_listen_mirror',
                    output='screen',
                    condition=None,
                    parameters=[{
                        'scene_file':    LaunchConfiguration('scene_file'),
                        'headless':      LaunchConfiguration('headless'),
                        'listen_mode':   True,
                        'listen_topic':  '/joint_states',
                        'enable_real':   False,
                    }],
                ),
            ],
        ),

        # 4) 真实 CTAG2F90D 夹爪节点：订阅 /joint_move_to 的夹爪字段，
        #    也支持 /ctag2f90d/open、/ctag2f90d/close、/ctag2f90d/target_position。
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='my_robot_mujoco_sim',
                    executable='ctag2f90d_gripper_node',
                    name='ctag2f90d_gripper_node',
                    output='screen',
                    condition=IfCondition(LaunchConfiguration('enable_gripper')),
                    parameters=[{
                        'controller_module_path': LaunchConfiguration('gripper_module_path'),
                        'arm_ip': LaunchConfiguration('real_ip'),
                        'arm_port': LaunchConfiguration('real_port'),
                        'device_id': LaunchConfiguration('gripper_device_id'),
                        'baudrate': LaunchConfiguration('gripper_baudrate'),
                        'follow_joint_move_to': True,
                        'default_speed': LaunchConfiguration('gripper_default_speed'),
                        'default_force': LaunchConfiguration('gripper_default_force'),
                    }],
                ),
            ],
        ),
    ])
