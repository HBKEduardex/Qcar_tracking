"""
MPPI Tracking Launch — Replacement for qcar2_nav2_only_launch.py
=================================================================
Launches the custom MPPI tracking controller instead of Nav2.
The MPPI node publishes to /nav2/motor_cmd (same topic the mixer reads),
so the full safety chain (LIDAR, person, traffic) remains active.

Usage:
  ros2 launch qcar2_teleop qcar2_mppi_tracking_launch.py

To fall back to Nav2 (original behavior):
  ros2 launch qcar2_teleop qcar2_nav2_only_launch.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('qcar2_teleop')

    # Default config paths
    mppi_config_path = os.path.join(pkg_dir, 'config', 'mppi_tracking_params.yaml')

    # ── Launch arguments ────────────────────────────────────────────────
    declare_config = DeclareLaunchArgument(
        'mppi_config',
        default_value=mppi_config_path,
        description='Path to MPPI tracking parameters YAML')

    mppi_config = LaunchConfiguration('mppi_config')

    # ── MPPI Tracking Node ──────────────────────────────────────────────
    mppi_node = Node(
        package='qcar2_teleop',
        executable='mppi_tracking_node',
        name='mppi_tracking_node',
        output='screen',
        parameters=[mppi_config],
    )

    # ── Build launch description ────────────────────────────────────────
    ld = LaunchDescription()
    ld.add_action(declare_config)
    ld.add_action(mppi_node)

    return ld
