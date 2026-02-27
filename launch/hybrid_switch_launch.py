#!/usr/bin/env python3
"""
Hybrid Switch Launch — Nav2 + Lane Tracking PID System

Brings up:
  1. yellow_line_follower_controller — PID lane following (publish_motor_cmd=false)
  2. hybrid_switch_controller        — mode switching controller
  3. bridge_monitor                   — terminal dashboard

Usage:
  ros2 launch qcar2_teleop hybrid_switch_launch.py
  ros2 launch qcar2_teleop hybrid_switch_launch.py yaw_error_threshold:=0.6
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── Package paths ───────────────────────────────────────────────────────
    teleop_dir = get_package_share_directory('qcar2_teleop')
    tracking_config = os.path.join(teleop_dir, 'config', 'qcar2_tracking_params.yaml')

    # ── Launch arguments ────────────────────────────────────────────────────
    declares = [
        DeclareLaunchArgument(
            'yaw_error_threshold', default_value='0.5',
            description='Yaw error (rad) above which Nav2 is used instead of PID',
        ),
        DeclareLaunchArgument(
            'goal_tolerance', default_value='0.3',
            description='Distance (m) to consider a goal reached',
        ),
        DeclareLaunchArgument(
            'retry_limit', default_value='5',
            description='Max Nav2 retries before skipping a goal',
        ),
        DeclareLaunchArgument(
            'rate_hz', default_value='20.0',
            description='Control loop frequency (Hz)',
        ),
        DeclareLaunchArgument(
            'map_frame', default_value='map',
            description='TF map frame',
        ),
        DeclareLaunchArgument(
            'base_frame', default_value='base_link',
            description='TF base frame of the robot',
        ),
        DeclareLaunchArgument(
            'mission_goals_topic', default_value='/mission_goals',
            description='Topic for receiving mission goals from the planner',
        ),
        DeclareLaunchArgument(
            'motor_cmd_topic', default_value='/qcar2_motor_speed_cmd',
            description='Topic for publishing final motor commands',
        ),
        DeclareLaunchArgument(
            'lane_cmd_topic', default_value='/lane/motor_cmd',
            description='Topic for receiving PID lane-following commands',
        ),
        DeclareLaunchArgument(
            'tracking_config', default_value=tracking_config,
            description='Config YAML for lane following nodes',
        ),
        DeclareLaunchArgument(
            'enable_monitor', default_value='true',
            description='Launch the bridge monitor node',
        ),
    ]

    # ── Configurations ──────────────────────────────────────────────────────
    yaw_thresh = LaunchConfiguration('yaw_error_threshold')
    goal_tol = LaunchConfiguration('goal_tolerance')
    retry_lim = LaunchConfiguration('retry_limit')
    rate = LaunchConfiguration('rate_hz')
    map_frame = LaunchConfiguration('map_frame')
    base_frame = LaunchConfiguration('base_frame')
    mission_topic = LaunchConfiguration('mission_goals_topic')
    motor_topic = LaunchConfiguration('motor_cmd_topic')
    lane_topic = LaunchConfiguration('lane_cmd_topic')
    config = LaunchConfiguration('tracking_config')

    # ═══════════════════════════════════════════════════════
    # 1) Yellow Line Follower Controller (PID)
    #    - publish_motor_cmd=false → only publishes /lane/motor_cmd
    #    - hybrid_switch_controller decides when to use it
    # ═══════════════════════════════════════════════════════
    pid_controller = Node(
        package='qcar2_teleop',
        executable='yellow_line_follower_controller',
        name='yellow_line_follower_controller',
        output='screen',
        parameters=[
            config,
            {
                'publish_motor_cmd': False,
                'send_speed_in_motor_cmd': True,
            }
        ],
        emulate_tty=True,
    )

    # ═══════════════════════════════════════════════════════
    # 2) Hybrid Switch Controller (NEW)
    # ═══════════════════════════════════════════════════════
    hybrid_controller = Node(
        package='qcar2_teleop',
        executable='hybrid_switch_controller',
        name='hybrid_switch_controller',
        output='screen',
        parameters=[{
            'yaw_error_threshold': yaw_thresh,
            'goal_tolerance': goal_tol,
            'retry_limit': retry_lim,
            'rate_hz': rate,
            'map_frame': map_frame,
            'base_frame': base_frame,
            'mission_goals_topic': mission_topic,
            'motor_cmd_topic': motor_topic,
            'lane_cmd_topic': lane_topic,
        }],
        emulate_tty=True,
    )

    # ═══════════════════════════════════════════════════════
    # 3) Bridge Monitor (optional)
    # ═══════════════════════════════════════════════════════
    bridge_monitor = Node(
        package='qcar2_teleop',
        executable='bridge_monitor',
        name='bridge_monitor',
        output='screen',
        emulate_tty=True,
    )

    # ── Build LaunchDescription ─────────────────────────────────────────────
    ld = LaunchDescription()

    for d in declares:
        ld.add_action(d)

    ld.add_action(pid_controller)
    ld.add_action(hybrid_controller)
    ld.add_action(bridge_monitor)

    return ld
