"""
Hybrid Nav2 + Lane Following Launch File (Combined)

Brings up:
  1. Nav2 stack (planner, controller, bt_navigator) — for path planning & RViz goals
  2. AMCL (localization)
  3. Lane following pipeline: color_segmentation + yellow_line_position
  4. (Optional) nav2_lane_bridge for hybrid control

Note:
  - Cartographer SLAM is already running from qcar2_LaneMapping-ACC (NOT launched here)
  - qcar2_to_lidar_tf is already running from qcar2_LaneMapping-ACC cartographer_mapping.launch
  - cartographer_occupancy_grid_node is already running from qcar2_LaneMapping-ACC
  - QCar2 hardware/virtual setup should be running separately (qcar2_nodex/qcar2_virtual_launch.py)
  - NO nav2_qcar2_converter → Nav2 plans routes but does NOT send motor commands
  - Lane following runs in parallel for visual lane detection
  - Bridge node fuses Nav2 velocity with lane-following steering

Usage:
  ros2 launch qcar2_teleop qcar2_hybrid_nav_launch.py
  ros2 launch qcar2_teleop qcar2_hybrid_nav_launch.py enable_bridge:=true
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, GroupAction,
    IncludeLaunchDescription, SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    # ─── Package directories ───
    qcar2_dir = get_package_share_directory('qcar2_nodes')
    teleop_dir = get_package_share_directory('qcar2_teleop')
    nav2_dir = get_package_share_directory('nav2_bringup')
    nav2_launch_dir = os.path.join(nav2_dir, 'launch')

    # ─── Launch configs ───
    namespace = LaunchConfiguration('namespace')
    use_namespace = LaunchConfiguration('use_namespace')
    slam = LaunchConfiguration('slam')
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    autostart = LaunchConfiguration('autostart')
    use_composition = LaunchConfiguration('use_composition')
    use_respawn = LaunchConfiguration('use_respawn')
    log_level = LaunchConfiguration('log_level')
    enable_bridge = LaunchConfiguration('enable_bridge')
    config_file = LaunchConfiguration('config_file')
    config_file_pid = LaunchConfiguration('config_file_pid')

    # ─── Declare arguments ───
    declares = [
        DeclareLaunchArgument('namespace', default_value=''),
        DeclareLaunchArgument('use_namespace', default_value='false'),
        DeclareLaunchArgument('slam', default_value='False'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(qcar2_dir, 'config', 'qcar2_slam_and_nav_virtual.yaml'),
        ),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument('use_composition', default_value='True'),
        DeclareLaunchArgument('use_respawn', default_value='False'),
        DeclareLaunchArgument('log_level', default_value='info'),
        DeclareLaunchArgument(
            'enable_bridge', default_value='false',
            description='Enable nav2_lane_bridge hybrid controller',
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value=os.path.join(teleop_dir, 'config', 'qcar2_tracking_params.yaml'),
            description='Full path to the config file to use',
        ),
        DeclareLaunchArgument(
            'config_file_pid',
            default_value=os.path.join(teleop_dir, 'config', 'pid_tunedv3.yaml'),
            description='Full path to the PID config file to use',
        ),
    ]

    # ─── YAML rewrite for Nav2 ───
    remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]
    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key=namespace,
        param_rewrites={'use_sim_time': use_sim_time},
        convert_types=True,
    )

    # ═══════════════════════════════════════════════════════
    # 1) NAV2 (planner + controller + bt_navigator) — NO converter
    # ═══════════════════════════════════════════════════════
    nav2_bringup_group = GroupAction([
        PushRosNamespace(
            condition=IfCondition(use_namespace),
            namespace=namespace,
        ),
        Node(
            condition=IfCondition(use_composition),
            name='nav2_container',
            package='rclcpp_components',
            executable='component_container_isolated',
            parameters=[configured_params, {'autostart': autostart}],
            arguments=['--ros-args', '--log-level', log_level],
            remappings=remappings,
            output='screen',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_launch_dir, 'slam_launch.py')
            ),
            condition=IfCondition(slam),
            launch_arguments={
                'namespace': namespace,
                'use_sim_time': use_sim_time,
                'autostart': autostart,
                'use_respawn': use_respawn,
                'params_file': params_file,
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_launch_dir, 'navigation_launch.py')
            ),
            launch_arguments={
                'namespace': namespace,
                'use_sim_time': use_sim_time,
                'autostart': autostart,
                'params_file': params_file,
                'use_composition': use_composition,
                'use_respawn': use_respawn,
                'container_name': 'nav2_container',
            }.items(),
        ),
    ])

    # AMCL (localization)
    amcl_node = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        parameters=[
            {'initial_pose': {'x': 0.0, 'y': 0.0, 'theta': 0.0}},
            {'map_topic': '/map'},
            {'scan_topic': '/scan'},
            {'odom_topic': '/odom'},
        ],
    )

    # ═══════════════════════════════════════════════════════
    # 2) LANE FOLLOWING PIPELINE (from qcar2_tracking_launch.py)
    # ═══════════════════════════════════════════════════════
    # 1. Color Segmentation Node
    color_segmentation_node = Node(
        package='qcar2_laneseg_acc',
        executable='color_segmentation_node.py',
        name='color_segmentation_node',
        output='screen',
        parameters=[config_file],
    )

    # 2. Yellow Line Position Node
    yellow_line_position_node = Node(
        package='qcar2_teleop',
        executable='yellow_line_position_node',
        name='yellow_line_position_node',
        output='screen',
        parameters=[config_file],
    )

    # ═══════════════════════════════════════════════════════
    # 3) HYBRID BRIDGE (optional — enable with enable_bridge:=true)
    # ═══════════════════════════════════════════════════════
    nav2_lane_bridge_node = Node(
        condition=IfCondition(enable_bridge),
        package='qcar2_teleop',
        executable='nav2_lane_bridge',
        name='nav2_lane_bridge',
        output='screen',
        parameters=[config_file],
    )

    # ═══════════════════════════════════════════════════════
    # Build LaunchDescription
    # ═══════════════════════════════════════════════════════
    ld = LaunchDescription()

    # Environment
    ld.add_action(SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'))

    # Declare all arguments
    for d in declares:
        ld.add_action(d)

    # 1) Nav2 (NO converter — planning only)
    ld.add_action(nav2_bringup_group)
    ld.add_action(amcl_node)

    # 2) Lane following
    ld.add_action(color_segmentation_node)
    ld.add_action(yellow_line_position_node)

    # 3) Bridge (optional)
    ld.add_action(nav2_lane_bridge_node)

    return ld
