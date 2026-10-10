cmake_minimum_required(VERSION 3.8)
project(so101_mirror)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# python module (kinematics, shared helpers)
ament_python_install_package(${PROJECT_NAME})

# nodes: `ros2 run so101_mirror <script>.py`
install(PROGRAMS
  scripts/ball_pose_publisher.py
  scripts/ball_sweep.py
  scripts/ik_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY description assets worlds launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
