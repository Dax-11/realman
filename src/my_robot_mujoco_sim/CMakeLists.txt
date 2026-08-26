cmake_minimum_required(VERSION 3.8)
project(my_robot_mujoco_sim)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# 安装 Python 包（节点代码）
ament_python_install_package(${PROJECT_NAME})

# 安装真臂 SDK Python 包（含 libs/linux_x86/libapi_c.so 等本地库）
ament_python_install_package(Robotic_Arm)

# 安装资源文件
install(
  DIRECTORY launch my_robot_mujoco_sim/models
  DESTINATION share/${PROJECT_NAME}
)

# 注册可执行入口
install(
  PROGRAMS
    my_robot_mujoco_sim/mujoco_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME mujoco_node
)

install(
  PROGRAMS
    my_robot_mujoco_sim/mujoco_real_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME mujoco_real_node
)


install(
  PROGRAMS
    my_robot_mujoco_sim/thyroid_scan/ros2_contact_scan_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME ros2_contact_scan_node
  OPTIONAL
)

install(
  PROGRAMS
    my_robot_mujoco_sim/thyroid_scan/ros2_scan_policy_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME ros2_scan_policy_node
  OPTIONAL
)

install(
  PROGRAMS
    my_robot_mujoco_sim/ros2_task_recorder_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME ros2_task_recorder_node
  OPTIONAL
)

install(
  PROGRAMS
    my_robot_mujoco_sim/ctag2f90d_gripper_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME ctag2f90d_gripper_node
)

install(
  PROGRAMS
    my_robot_mujoco_sim/multi_view_scan_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME multi_view_scan_node
)

# 新增：视觉与控制节点入口（已去除 .py 后缀）
install(
  PROGRAMS
    my_robot_mujoco_sim/thyroid_detector_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME thyroid_detector_node
)

install(
  PROGRAMS
    my_robot_mujoco_sim/thyroid_arm_control_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME thyroid_arm_control_node
)

ament_package()
