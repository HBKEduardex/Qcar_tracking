import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_qcar_tracking = get_package_share_directory('qcar2_teleop')
    
    # Path to config file
    config_file_path = os.path.join(
        pkg_qcar_tracking,
        'config',
        'qcar2_tracking_params.yaml'
    )
    config_file_path_pid = os.path.join(
        pkg_qcar_tracking,
        'config',
        'pid_tunedv3.yaml'
    )

    # Launch configuration variables
    config_file = LaunchConfiguration('config_file')
    config_file_pid = LaunchConfiguration('config_file_pid')

    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file',
        default_value=config_file_path,
        description='Full path to the config file to use'
    )

    declare_config_file_pid_cmd = DeclareLaunchArgument(
        'config_file_pid',
        default_value=config_file_path_pid,
        description='Full path to the config file to use'
    )

    # 1. Color Segmentation Node
    # ros2 run qcar2_laneseg_acc color_segmentation_node.py
    color_segmentation_node = Node(
        package='qcar2_laneseg_acc',
        executable='color_segmentation_node.py',
        name='color_segmentation_node',
        output='screen',
        parameters=[config_file]
    )

    # 2. Yellow Line Position Node
    # ros2 run qcar2_teleop yellow_line_position_node
    yellow_line_position_node = Node(
        package='qcar2_teleop',
        executable='yellow_line_position_node',
        name='yellow_line_position_node',
        output='screen',
        parameters=[config_file]
    )

    # 3. Yellow Line Follower Controller
    # ros2 run qcar2_teleop yellow_line_follower_controller
    yellow_line_follower_controller = Node(
        package='qcar2_teleop',
        executable='yellow_line_follower_controller',
        name='yellow_line_follower_controller',
        output='screen',
        parameters=[config_file, config_file_pid]
    )

    # 4. Controller Plotter Node
    # ros2 run qcar2_teleop controller_plotter_node.py
    # Note: setup.py entry point says 'controller_plotter_node = ...' 
    # but usually if it's a script it might just be the script name if installed as data.
    # checking setup.py again: 'controller_plotter_node = qcar2_teleop.controller_plotter_node:main'
    # So the executable name declared in console_scripts is 'controller_plotter_node' (without .py)
    # BUT the user request said: "lanzar @[qcar2_teleop/controller_plotter_node.py]" 
    # and in setup.py it is 'controller_plotter_node = ...'. 
    # Usually ros2 run uses the console_script name. 
    # However, sometimes people convert scripts to executables with .py extension if they use correct setup.
    # The setup.py shows: 'controller_plotter_node = qcar2_teleop.controller_plotter_node:main'
    # SO the executable name is 'controller_plotter_node'.
    # I will use 'controller_plotter_node' for the executable name in Node.
    
    controller_plotter_node = Node(
        package='qcar2_teleop',
        executable='controller_plotter_node',
        name='controller_plotter',
        output='screen',
        parameters=[config_file]
    )

    ld = LaunchDescription()
    ld.add_action(declare_config_file_cmd)
    ld.add_action(declare_config_file_pid_cmd)
    ld.add_action(color_segmentation_node)
    ld.add_action(yellow_line_position_node)
    ld.add_action(yellow_line_follower_controller)
    ld.add_action(controller_plotter_node)

    return ld
