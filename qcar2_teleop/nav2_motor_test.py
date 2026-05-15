#!/usr/bin/env python3
"""
Nav2-Only Controller (Test Node)

Everything the hybrid controller does, MINUS the PID branch.
Always uses Nav2 — bridges /cmd_vel_nav → motor commands.

Use this to debug why Nav2 crashes or stops working.

Flow:
  1. Planner sends semi-goals on /mission_goals
  2. This node sends each goal to /goal_pose for Nav2
  3. Nav2 plans + publishes /cmd_vel_nav
  4. This node bridges cmd_vel → motor commands
  5. When goal reached (distance < tolerance), advances to next goal
  6. Re-sends next goal to /goal_pose
  7. Repeat until all goals done

Extensive logging at every step for debugging.

Usage:
  # Terminal 1: Hardware + SLAM + Nav2
  ros2 launch qcar2_teleop mamalaunch.py

  # Terminal 2: Planner
  ros2 launch qcar2_planner qcar2_planner_server.launch.py

  # Terminal 3: This test node (replaces hybrid controller)
  ros2 run qcar2_teleop nav2_motor_test

  # RViz: 2D Goal Pose → planner → mission_goals → this node

Subscribes:
    /mission_goals     (PoseStamped)  — from planner
    /cmd_vel_nav       (Twist)        — from Nav2

Publishes:
    /goal_pose              (PoseStamped)      — for Nav2
    /qcar2_motor_speed_cmd  (MotorCommands)    — motor commands
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from qcar2_interfaces.msg import MotorCommands
import tf2_ros
from tf2_ros import Buffer, TransformListener


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def quat_to_yaw(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class Nav2MotorTest(Node):
    def __init__(self):
        super().__init__('nav2_motor_test')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('steer_gain', 1.0)
        self.declare_parameter('speed_scale', 1.0)
        self.declare_parameter('max_angle', 0.45)
        self.declare_parameter('max_speed', 0.30)
        self.declare_parameter('timeout', 0.5)
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('goal_tolerance', 0.3)
        self.declare_parameter('map_frame', 'pgm_map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('retry_interval', 3.0)    # seconds between retries

        self.steer_gain = float(self.get_parameter('steer_gain').value)
        self.speed_scale = float(self.get_parameter('speed_scale').value)
        self.max_angle = float(self.get_parameter('max_angle').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.timeout = float(self.get_parameter('timeout').value)
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.retry_interval = float(self.get_parameter('retry_interval').value)

        # ── State ───────────────────────────────────────────────────────────
        self.mission_goals = []
        self.current_goal_idx = 0
        self.goal_sent_to_nav2 = False

        # Nav2 cmd_vel
        self.linear_x = 0.0
        self.angular_z = 0.0
        self.last_cmd_time = self.get_clock().now()
        self.cmd_vel_count = 0

        # Auto-retry
        self.last_retry_time = self.get_clock().now()
        self.retry_count = 0

        # ── TF2 ─────────────────────────────────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Subs ────────────────────────────────────────────────────────────
        self.create_subscription(
            PoseStamped, '/mission_goals', self._cb_mission_goal, 10
        )
        self.create_subscription(
            Twist, '/cmd_vel_nav', self._cb_cmd_vel, 10
        )

        # ── Pubs ────────────────────────────────────────────────────────────
        self.motor_pub = self.create_publisher(
            MotorCommands, '/qcar2_motor_speed_cmd', 10
        )
        self.goal_pose_pub = self.create_publisher(
            PoseStamped, '/goal_pose', 10
        )

        # ── Timer ───────────────────────────────────────────────────────────
        period = 1.0 / max(1.0, self.rate_hz)
        self.create_timer(period, self._control_loop)

        self.get_logger().info(
            '═══════════════════════════════════════════════════\n'
            '  NAV2-ONLY TEST CONTROLLER\n'
            '  /mission_goals → /goal_pose → /cmd_vel_nav → motors\n'
            f'  steer_gain={self.steer_gain}, speed_scale={self.speed_scale}\n'
            f'  goal_tolerance={self.goal_tolerance}m\n'
            '  Send 2D Goal Pose from RViz (planner must be running)\n'
            '═══════════════════════════════════════════════════'
        )

    # ── Mission goals ───────────────────────────────────────────────────────
    def _cb_mission_goal(self, msg: PoseStamped):
        if self.current_goal_idx >= len(self.mission_goals):
            self.mission_goals = []
            self.current_goal_idx = 0
            self.goal_sent_to_nav2 = False

        self.mission_goals.append(msg)
        total = len(self.mission_goals)

        self.get_logger().info(
            f'📍 Mission goal {total} received: '
            f'({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})'
        )

    # ── Nav2 cmd_vel ────────────────────────────────────────────────────────
    def _cb_cmd_vel(self, msg: Twist):
        self.linear_x = msg.linear.x
        self.angular_z = msg.angular.z
        self.last_cmd_time = self.get_clock().now()
        self.cmd_vel_count += 1

    # ── Main control loop ───────────────────────────────────────────────────
    def _control_loop(self):
        # No goals → stop
        if not self.mission_goals or self.current_goal_idx >= len(self.mission_goals):
            self._publish_motor(0.0, 0.0)
            return

        # Get robot pose
        robot_pose = self._get_robot_pose()
        if robot_pose is None:
            return

        rx, ry, ryaw = robot_pose

        # Current goal
        goal = self.mission_goals[self.current_goal_idx]
        gx = goal.pose.position.x
        gy = goal.pose.position.y

        # Send goal to Nav2 if not yet sent
        if not self.goal_sent_to_nav2:
            self._send_goal(goal)
            self.goal_sent_to_nav2 = True

        # Check distance
        dist = math.sqrt((gx - rx)**2 + (gy - ry)**2)

        # Goal reached?
        if dist <= self.goal_tolerance:
            self.get_logger().info(
                f'✅ Goal {self.current_goal_idx + 1}/{len(self.mission_goals)} '
                f'REACHED (dist={dist:.3f}m)'
            )
            self.current_goal_idx += 1
            self.goal_sent_to_nav2 = False

            if self.current_goal_idx >= len(self.mission_goals):
                self.get_logger().info('🏁 ALL goals complete! Stopping.')
                self._publish_motor(0.0, 0.0)
                return

            # Send next goal
            next_goal = self.mission_goals[self.current_goal_idx]
            self._send_goal(next_goal)
            self.goal_sent_to_nav2 = True
            return

        # Bridge cmd_vel → motors
        now = self.get_clock().now()
        dt_cmd = (now - self.last_cmd_time).nanoseconds * 1e-9

        if dt_cmd > self.timeout:
            # Nav2 stopped sending cmd_vel — AUTO RETRY
            self._publish_motor(0.0, 0.0)

            # Check if enough time has passed since last retry
            dt_retry = (now - self.last_retry_time).nanoseconds * 1e-9
            if dt_retry >= self.retry_interval:
                self.retry_count += 1
                self.last_retry_time = now
                self.get_logger().warn(
                    f'🔁 RETRY #{self.retry_count}: Nav2 cmd_vel timeout '
                    f'({dt_cmd:.1f}s) | dist={dist:.3f}m | '
                    f'Re-sending Goal {self.current_goal_idx + 1}/'
                    f'{len(self.mission_goals)} to /goal_pose'
                )
                self._send_goal(goal)
            elif self.cmd_vel_count > 0:
                self.get_logger().warn(
                    f'⚠️ Nav2 cmd_vel TIMEOUT ({dt_cmd:.1f}s) | '
                    f'dist={dist:.3f}m | '
                    f'retry #{self.retry_count} | '
                    f'next retry in {self.retry_interval - dt_retry:.1f}s',
                    throttle_duration_sec=2.0
                )
            return

        # Active — bridge to motors
        steering = clamp(
            self.steer_gain * self.angular_z,
            -self.max_angle, self.max_angle
        )
        speed = clamp(
            self.speed_scale * self.linear_x,
            -self.max_speed, self.max_speed
        )
        self._publish_motor(steering, speed)

        # Periodic status (every 2 seconds)
        if self.cmd_vel_count > 0 and self.cmd_vel_count % 40 == 0:
            angle_to = math.atan2(gy - ry, gx - rx)
            yaw_err = math.atan2(math.sin(angle_to - ryaw), math.cos(angle_to - ryaw))
            self.get_logger().info(
                f'🔧 Goal {self.current_goal_idx + 1}/{len(self.mission_goals)} | '
                f'dist={dist:.3f}m | '
                f'yaw_err={yaw_err:+.2f}rad ({math.degrees(yaw_err):+.1f}°) | '
                f'steer={steering:+.3f} speed={speed:+.3f} | '
                f'nav2: lin={self.linear_x:+.3f} ang={self.angular_z:+.3f}'
            )

    # ── Send goal to Nav2 ───────────────────────────────────────────────────
    def _send_goal(self, goal_msg: PoseStamped):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.map_frame
        pose_msg.pose = goal_msg.pose
        self.goal_pose_pub.publish(pose_msg)

        gx = goal_msg.pose.position.x
        gy = goal_msg.pose.position.y
        gyaw = quat_to_yaw(goal_msg.pose.orientation)
        self.get_logger().info(
            f'📤 SENT to /goal_pose: Goal {self.current_goal_idx + 1}/'
            f'{len(self.mission_goals)} → '
            f'({gx:.2f}, {gy:.2f}, yaw={gyaw:+.2f}rad)'
        )

    # ── Motor publishing ────────────────────────────────────────────────────
    def _publish_motor(self, steering, speed):
        msg = MotorCommands()
        msg.motor_names = ['steering_angle', 'motor_throttle']
        msg.values = [float(steering), float(speed)]
        self.motor_pub.publish(msg)

    # ── TF ──────────────────────────────────────────────────────────────────
    def _get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time()
            )
            x = t.transform.translation.x
            y = t.transform.translation.y
            yaw = quat_to_yaw(t.transform.rotation)
            return x, y, yaw
        except Exception:
            return None


def main(args=None):
    rclpy.init(args=args)
    node = Nav2MotorTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_motor(0.0, 0.0)
        node.get_logger().info('Motors stopped.')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
