import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription, SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml
from launch_ros.actions import SetRemap

def generate_launch_description():
    # Directories
    bringup_dir = get_package_share_directory('qcar2_nodes')
    nav2_dir = get_package_share_directory('nav2_bringup')
    launch_dir = os.path.join(nav2_dir, 'launch')

    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    autostart = LaunchConfiguration('autostart')
    use_composition = LaunchConfiguration('use_composition')
    log_level = LaunchConfiguration('log_level')

    # Create our own temporary YAML files that include substitutions
    param_substitutions = {
        'use_sim_time': use_sim_time}

    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key='',
        param_rewrites=param_substitutions,
        convert_types=True)

    # Declarations
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true')

    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(bringup_dir, 'config', 'qcar2_slam_and_nav.yaml'),
        description='Full path to the ROS2 parameters file to use for all launched nodes')

    declare_autostart_cmd = DeclareLaunchArgument(
        'autostart', default_value='true',
        description='Automatically startup the nav2 stack')

    declare_use_composition_cmd = DeclareLaunchArgument(
        'use_composition', default_value='True',
        description='Whether to use composed bringup')

    declare_log_level_cmd = DeclareLaunchArgument(
        'log_level', default_value='info',
        description='log level')

    # ─────────────────────────────────────────────────────────────────
    # Nav2 bringup group
    # 
    # IMPORTANTE sobre los remapeos de cmd_vel:
    # navigation_launch.py de Nav2 ya tiene remapeos internos:
    #   - controller_server: cmd_vel → cmd_vel_nav
    #   - velocity_smoother: lee cmd_vel_nav, publica cmd_vel_smoothed → cmd_vel
    #
    # Cadena final:
    #   MPPI → cmd_vel_nav → velocity_smoother → cmd_vel → converter C++
    #
    # Por lo tanto NO debemos remapear cmd_vel aquí dentro, porque
    # se apilaría con el remapeo interno y rompería la cadena.
    # ─────────────────────────────────────────────────────────────────
    bringup_cmd_group = GroupAction([
        # Remapeo de mapa: Nav2 lee /map → redirigir al overlay PGM
        SetRemap(src='/map', dst='/planner_occupancy'),

        # Remapeo de goals: Nav2 escucha goal_pose → redirigir al planner
        SetRemap(src='goal_pose', dst='/mission_goals'),

        # NO remapear cmd_vel aquí — Nav2 ya tiene su cadena interna

        Node(
            condition=IfCondition(use_composition),
            name='nav2_container',
            package='rclcpp_components',
            executable='component_container_isolated',
            parameters=[configured_params, {'autostart': autostart}],
            arguments=['--ros-args', '--log-level', log_level],
            output='screen'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(launch_dir, 'navigation_launch.py')),
            launch_arguments={'use_sim_time': use_sim_time,
                              'autostart': autostart,
                              'params_file': params_file,
                              'use_composition': use_composition,
                              'container_name': 'nav2_container'}.items()),
    ])
    
    # ─────────────────────────────────────────────────────────────────
    # Converter: escucha la salida FINAL de Nav2 (cmd_vel, que ya
    # viene suavizada por velocity_smoother) y la convierte a
    # MotorCommands para el hardware del QCar2.
    # ─────────────────────────────────────────────────────────────────
    qcar2_nav2_converter = Node(
        package='qcar2_nodex',
        executable='nav2_qcar2_converter',
        name='nav2_qcar2_converter',
        output='screen',
        # El converter originalmente escucha /cmd_vel_nav.
        # Ahora la salida final de Nav2 (post-smoother) es /cmd_vel.
        remappings=[('/cmd_vel_nav', '/cmd_vel')]
    )

    # Create the launch description and populate
    ld = LaunchDescription()

    # Set environment variables
    ld.add_action(SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'))

    # Declare the launch options
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_autostart_cmd)
    ld.add_action(declare_use_composition_cmd)
    ld.add_action(declare_log_level_cmd)

    # Add the actions to launch navigation (remapeos están DENTRO del grupo)
    ld.add_action(bringup_cmd_group)
    
    # Add converter for physical hardware
    ld.add_action(qcar2_nav2_converter)

    return ld
