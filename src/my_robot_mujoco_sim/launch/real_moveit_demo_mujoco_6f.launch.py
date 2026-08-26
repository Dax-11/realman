from moveit_configs_utils import MoveItConfigsBuilder

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from moveit_configs_utils.launch_utils import add_debuggable_node, DeclareBooleanLaunchArg
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # 旧方案保留作参考：直接使用官方 6F 版描述文件。
    # moveit_config = (
    #     MoveItConfigsBuilder("rm_75_description", package_name="rm_75_config")
    #     .robot_description(
    #         file_path="config/rm_75_6fb_description.urdf.xacro",
    #         mappings={"link7_type": "Link7_6f"},
    #     )
    #     .to_moveit_configs()
    # )

    # 新方案：仍然沿用官方 RM75 运动学和 MoveIt 配置，
    # 但把 robot_description 切到“MuJoCo 对齐版” xacro。
    moveit_config = (
        MoveItConfigsBuilder("rm_75_description", package_name="rm_75_config")
        .robot_description(
            file_path="config/rm_75_mujoco_aligned_6f.urdf.xacro",
            mappings={"link7_type": "Link7_6f"},
        )
        .to_moveit_configs()
    )

    ld = LaunchDescription()
    my_generate_move_group_launch(ld, moveit_config)
    my_generate_moveit_rviz_launch(ld, moveit_config)
    return ld


def my_generate_move_group_launch(ld, moveit_config):
    ld.add_action(DeclareBooleanLaunchArg("debug", default_value=False))
    ld.add_action(DeclareBooleanLaunchArg("allow_trajectory_execution", default_value=True))
    ld.add_action(DeclareBooleanLaunchArg("publish_monitored_planning_scene", default_value=True))
    ld.add_action(DeclareLaunchArgument("capabilities", default_value=""))
    ld.add_action(DeclareLaunchArgument("disable_capabilities", default_value=""))
    ld.add_action(DeclareBooleanLaunchArg("monitor_dynamics", default_value=False))

    move_group_configuration = {
        "publish_robot_description_semantic": True,
        "allow_trajectory_execution": LaunchConfiguration("allow_trajectory_execution"),
        "capabilities": ParameterValue(LaunchConfiguration("capabilities"), value_type=str),
        "disable_capabilities": ParameterValue(LaunchConfiguration("disable_capabilities"), value_type=str),
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
        "monitor_dynamics": False,
    }
    trajectory_execution = {
        "moveit_manage_controllers": False,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.15,
    }

    move_group_params = [
        moveit_config.to_dict(),
        move_group_configuration,
        trajectory_execution,
    ]

    add_debuggable_node(
        ld,
        package="moveit_ros_move_group",
        executable="move_group",
        commands_file=str(moveit_config.package_path / "launch" / "gdb_settings.gdb"),
        output="screen",
        parameters=move_group_params,
        extra_debug_args=["--debug"],
        additional_env={"DISPLAY": ":0"},
    )
    return ld


def my_generate_moveit_rviz_launch(ld, moveit_config):
    ld.add_action(DeclareBooleanLaunchArg("debug", default_value=False))
    ld.add_action(
        DeclareLaunchArgument(
            "rviz_config",
            default_value=str(moveit_config.package_path / "config/moveit.rviz"),
        )
    )

    rviz_parameters = [
        moveit_config.robot_description,
        moveit_config.robot_description_semantic,
        moveit_config.planning_pipelines,
        moveit_config.robot_description_kinematics,
    ]

    add_debuggable_node(
        ld,
        package="rviz2",
        executable="rviz2",
        output="log",
        respawn=False,
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=rviz_parameters,
    )

    return ld
