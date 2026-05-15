"""
Launch file for the Pixel-Gated Hybrid Switch System.

Launches:
  1. yellow_line_follower_controller (PID, publish_motor_cmd=False)
  2. hybrid_switch_controller (pixel-gated FSM, the ONLY motor publisher)
  3. bridge_monitor (terminal display)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── Pixel gating params ─────────────────────────────────────────────
    args = [
        DeclareLaunchArgument('mask_topic', default_value='/lokita'),
        DeclareLaunchArgument('use_bottom_ratio', default_value='0.45'),
        DeclareLaunchArgument('yellow_low_threshold', default_value='0.04'),
        DeclareLaunchArgument('blue_high_threshold', default_value='0.6'),
        DeclareLaunchArgument('blue_exit_threshold', default_value='0.3'),
        DeclareLaunchArgument('activate_frames', default_value='3'),
        DeclareLaunchArgument('reacquire_frames', default_value='5'),
        DeclareLaunchArgument('cooldown_sec', default_value='3.0'),
        DeclareLaunchArgument('gate_mode', default_value='AND'),

        # Navigation
        DeclareLaunchArgument('goal_tolerance', default_value='0.3'),
        DeclareLaunchArgument('rate_hz', default_value='20.0'),
        DeclareLaunchArgument('map_frame', default_value='pgm_map'),
        DeclareLaunchArgument('base_frame', default_value='base_link'),
        DeclareLaunchArgument('mission_goals_topic', default_value='/mission_goals'),
        DeclareLaunchArgument('motor_cmd_topic', default_value='/qcar2_motor_speed_cmd'),
        DeclareLaunchArgument('lane_cmd_topic', default_value='/lane/motor_cmd'),
        DeclareLaunchArgument('goal_pose_topic', default_value='/goal_pose'),
        DeclareLaunchArgument('cmd_vel_topic', default_value='/cmd_vel_nav'),
        DeclareLaunchArgument('pid_speed_override', default_value='0.0'),

        # Nav2 bridging
        DeclareLaunchArgument('nav2_speed_scale', default_value='1.0'),
        DeclareLaunchArgument('max_angle', default_value='0.45'),
        DeclareLaunchArgument('max_speed', default_value='0.30'),
        DeclareLaunchArgument('nav2_timeout', default_value='0.5'),
        DeclareLaunchArgument('retry_interval', default_value='3.0'),
        DeclareLaunchArgument('yaw_error_threshold', default_value='0.5'),
    ]

    # ── PID Controller (output only, no motor publishing) ───────────────
    pid_node = Node(
        package='qcar2_teleop',
        executable='yellow_line_follower_controller',
        name='yellow_line_follower_controller',
        output='screen',
        parameters=[{
            'publish_motor_cmd': False,
            'motor_cmd_topic': LaunchConfiguration('lane_cmd_topic'),
        }],
    )

    # ── Hybrid Switch Controller (pixel-gated) ─────────────────────────
    hybrid_node = Node(
        package='qcar2_teleop',
        executable='hybrid_switch_controller',
        name='hybrid_switch_controller',
        output='screen',
        parameters=[{
            'mask_topic': LaunchConfiguration('mask_topic'),
            'use_bottom_ratio': LaunchConfiguration('use_bottom_ratio'),
            'yellow_low_threshold': LaunchConfiguration('yellow_low_threshold'),
            'blue_high_threshold': LaunchConfiguration('blue_high_threshold'),
            'blue_exit_threshold': LaunchConfiguration('blue_exit_threshold'),
            'activate_frames': LaunchConfiguration('activate_frames'),
            'reacquire_frames': LaunchConfiguration('reacquire_frames'),
            'cooldown_sec': LaunchConfiguration('cooldown_sec'),
            'gate_mode': LaunchConfiguration('gate_mode'),
            'goal_tolerance': LaunchConfiguration('goal_tolerance'),
            'rate_hz': LaunchConfiguration('rate_hz'),
            'map_frame': LaunchConfiguration('map_frame'),
            'base_frame': LaunchConfiguration('base_frame'),
            'mission_goals_topic': LaunchConfiguration('mission_goals_topic'),
            'motor_cmd_topic': LaunchConfiguration('motor_cmd_topic'),
            'lane_cmd_topic': LaunchConfiguration('lane_cmd_topic'),
            'goal_pose_topic': LaunchConfiguration('goal_pose_topic'),
            'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
            'pid_speed_override': LaunchConfiguration('pid_speed_override'),
            'nav2_speed_scale': LaunchConfiguration('nav2_speed_scale'),
            'max_angle': LaunchConfiguration('max_angle'),
            'max_speed': LaunchConfiguration('max_speed'),
            'nav2_timeout': LaunchConfiguration('nav2_timeout'),
            'retry_interval': LaunchConfiguration('retry_interval'),
            'yaw_error_threshold': LaunchConfiguration('yaw_error_threshold'),
        }],
    )

    # ── Bridge Monitor ──────────────────────────────────────────────────
    monitor_node = Node(
        package='qcar2_teleop',
        executable='bridge_monitor',
        name='bridge_monitor',
        output='screen',
    )

    return LaunchDescription(args + [pid_node, hybrid_node, monitor_node])
