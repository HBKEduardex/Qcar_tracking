#!/usr/bin/env python3
"""
Hybrid Switch Controller — Nav2 ↔ Lane Tracking PID

Built on the PROVEN nav2_motor_test pattern. Adds PID lane-following
as an alternative mode when the angular error to the goal is small.

Decision rule (angle TO current mission goal position):
    angle_to_goal = atan2(goal_y - robot_y, goal_x - robot_x)
    yaw_error = normalize(angle_to_goal - robot_yaw)

    if |yaw_error| > yaw_error_threshold → Nav2 (turning toward goal)
    else                                 → PID  (lane following)

NAV2 MODE: sends goal to /goal_pose, bridges /cmd_vel_nav → motors.
PID MODE:  forwards /lane/motor_cmd → motors.
AUTO-RETRY: if Nav2 stops sending cmd_vel, re-sends goal every retry_interval.

Subscribes:
    /mission_goals     (PoseStamped)   — from directional_planner_server
    /lane/motor_cmd    (MotorCommands) — from yellow_line_follower_controller
    /cmd_vel_nav       (Twist)         — from Nav2

Publishes:
    /goal_pose              (PoseStamped)   — for Nav2
    /qcar2_motor_speed_cmd  (MotorCommands) — final motor commands
    /hybrid/mode            (Float32)       — 1.0=PID, 0.0=NAV2, -1.0=STOPPED
    /hybrid/steering        (Float32)
    /hybrid/speed           (Float32)
    /hybrid/goal_index      (String)        — "3/6"
    /hybrid/yaw_error       (Float32)
    /hybrid/yaw_robot       (Float32)
    /hybrid/yaw_goal        (Float32)       — angle TO goal position
    /hybrid/yaw_threshold   (Float32)
    /hybrid/nav2_status     (String)
    /hybrid/goal_pose       (PoseStamped)
"""

import math
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32, String
from geometry_msgs.msg import PoseStamped, Twist
from qcar2_interfaces.msg import MotorCommands

import tf2_ros
from tf2_ros import Buffer, TransformListener


# Modes
MODE_STOPPED = -1.0
MODE_NAV2    =  0.0
MODE_PID     =  1.0


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def quat_to_yaw(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class HybridSwitchController(Node):

    def __init__(self):
        super().__init__('hybrid_switch_controller')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('yaw_error_threshold', 1.4)      # rad (~80°)
        self.declare_parameter('goal_tolerance', 0.3)
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('mission_goals_topic', '/mission_goals')
        self.declare_parameter('motor_cmd_topic', '/qcar2_motor_speed_cmd')
        self.declare_parameter('lane_cmd_topic', '/lane/motor_cmd')
        self.declare_parameter('goal_pose_topic', '/goal_pose')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel_nav')
        self.declare_parameter('pid_speed_override', 0.0)

        # Nav2 bridging
        self.declare_parameter('nav2_steer_gain', 1.0)
        self.declare_parameter('nav2_speed_scale', 1.0)
        self.declare_parameter('max_angle', 0.45)
        self.declare_parameter('max_speed', 0.30)
        self.declare_parameter('nav2_timeout', 0.5)
        self.declare_parameter('retry_interval', 3.0)

        # ── Read parameters ─────────────────────────────────────────────────
        self.yaw_error_threshold = float(self.get_parameter('yaw_error_threshold').value)
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        mission_topic = str(self.get_parameter('mission_goals_topic').value)
        motor_topic = str(self.get_parameter('motor_cmd_topic').value)
        lane_topic = str(self.get_parameter('lane_cmd_topic').value)
        goal_pose_topic = str(self.get_parameter('goal_pose_topic').value)
        cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.pid_speed_override = float(self.get_parameter('pid_speed_override').value)
        self.nav2_steer_gain = float(self.get_parameter('nav2_steer_gain').value)
        self.nav2_speed_scale = float(self.get_parameter('nav2_speed_scale').value)
        self.max_angle = float(self.get_parameter('max_angle').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.nav2_timeout = float(self.get_parameter('nav2_timeout').value)
        self.retry_interval = float(self.get_parameter('retry_interval').value)

        # ── State ───────────────────────────────────────────────────────────
        self.mission_goals = []
        self.current_goal_idx = 0
        self.current_mode = MODE_STOPPED
        self.goal_sent_to_nav2 = False

        # Nav2 cmd_vel
        self.nav2_linear_x = 0.0
        self.nav2_angular_z = 0.0
        self.last_cmd_vel_time = self.get_clock().now()

        # Lane PID
        self.lane_steering = 0.0
        self.lane_speed = 0.0
        self.last_lane_cmd_time = self.get_clock().now()

        # Yaw tracking
        self.yaw_robot = 0.0
        self.yaw_goal = 0.0
        self.yaw_error = 0.0

        # Active output
        self.active_steering = 0.0
        self.active_speed = 0.0
        self.nav2_status = 'IDLE'

        # Auto-retry
        self.last_retry_time = self.get_clock().now()
        self.retry_count = 0

        # ── TF2 ─────────────────────────────────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Subscribers ─────────────────────────────────────────────────────
        self.create_subscription(
            PoseStamped, mission_topic, self._cb_mission_goal, 10)
        self.create_subscription(
            MotorCommands, lane_topic, self._cb_lane_cmd, 10)
        self.create_subscription(
            Twist, cmd_vel_topic, self._cb_cmd_vel, 10)

        # ── Publishers ──────────────────────────────────────────────────────
        self.motor_pub = self.create_publisher(MotorCommands, motor_topic, 10)
        self.goal_pose_pub = self.create_publisher(PoseStamped, goal_pose_topic, 10)

        self.pub_mode = self.create_publisher(Float32, '/hybrid/mode', 10)
        self.pub_steering = self.create_publisher(Float32, '/hybrid/steering', 10)
        self.pub_speed = self.create_publisher(Float32, '/hybrid/speed', 10)
        self.pub_goal_index = self.create_publisher(String, '/hybrid/goal_index', 10)
        self.pub_yaw_error = self.create_publisher(Float32, '/hybrid/yaw_error', 10)
        self.pub_yaw_robot = self.create_publisher(Float32, '/hybrid/yaw_robot', 10)
        self.pub_yaw_goal = self.create_publisher(Float32, '/hybrid/yaw_goal', 10)
        self.pub_yaw_threshold = self.create_publisher(Float32, '/hybrid/yaw_threshold', 10)
        self.pub_nav2_status = self.create_publisher(String, '/hybrid/nav2_status', 10)
        self.pub_goal_pose = self.create_publisher(PoseStamped, '/hybrid/goal_pose', 10)

        # ── Timer ───────────────────────────────────────────────────────────
        period = 1.0 / max(1.0, self.rate_hz)
        self.create_timer(period, self._control_loop)

        self.get_logger().info(
            f'HybridSwitchController started — '
            f'yaw_thresh={self.yaw_error_threshold:.2f}rad '
            f'({math.degrees(self.yaw_error_threshold):.1f}°), '
            f'goal_tol={self.goal_tolerance:.2f}m, '
            f'retry_interval={self.retry_interval:.1f}s'
        )

    # =====================================================================
    # Callbacks
    # =====================================================================
    def _cb_mission_goal(self, msg: PoseStamped):
        """Accumulate mission goals from the planner."""
        if self.current_goal_idx >= len(self.mission_goals):
            self.mission_goals = []
            self.current_goal_idx = 0
            self.goal_sent_to_nav2 = False
            self.retry_count = 0

        self.mission_goals.append(msg)
        self.get_logger().info(
            f'📍 Mission goal {len(self.mission_goals)} received: '
            f'({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})'
        )

    def _cb_lane_cmd(self, msg: MotorCommands):
        """Store latest PID output."""
        self.last_lane_cmd_time = self.get_clock().now()
        for i, name in enumerate(msg.motor_names):
            if name == 'steering_angle' and i < len(msg.values):
                self.lane_steering = float(msg.values[i])
            elif name == 'motor_throttle' and i < len(msg.values):
                self.lane_speed = float(msg.values[i])

    def _cb_cmd_vel(self, msg: Twist):
        """Store Nav2 velocity commands."""
        self.nav2_linear_x = msg.linear.x
        self.nav2_angular_z = msg.angular.z
        self.last_cmd_vel_time = self.get_clock().now()

    # =====================================================================
    # Main control loop — same structure as nav2_motor_test + PID branch
    # =====================================================================
    def _control_loop(self):
        # ── No goals → STOPPED ──────────────────────────────────────────
        if not self.mission_goals or self.current_goal_idx >= len(self.mission_goals):
            self.current_mode = MODE_STOPPED
            self._publish_motor(0.0, 0.0)
            self._publish_debug()
            return

        # ── Get robot pose ──────────────────────────────────────────────
        robot_pose = self._get_robot_pose()
        if robot_pose is None:
            self._publish_debug()
            return

        rx, ry, ryaw = robot_pose
        self.yaw_robot = ryaw

        # ── Current goal ────────────────────────────────────────────────
        goal = self.mission_goals[self.current_goal_idx]
        gx = goal.pose.position.x
        gy = goal.pose.position.y

        # ── Yaw error = angle TO current goal position ──────────────────
        angle_to_goal = math.atan2(gy - ry, gx - rx)
        self.yaw_error = normalize_angle(angle_to_goal - ryaw)
        self.yaw_goal = angle_to_goal

        # ── Distance check ──────────────────────────────────────────────
        dist = math.sqrt((gx - rx)**2 + (gy - ry)**2)

        # ── Goal reached? ───────────────────────────────────────────────
        if dist <= self.goal_tolerance:
            self.get_logger().info(
                f'✅ Goal {self.current_goal_idx + 1}/'
                f'{len(self.mission_goals)} reached (dist={dist:.3f}m)'
            )
            self.current_goal_idx += 1
            self.goal_sent_to_nav2 = False
            self.retry_count = 0

            if self.current_goal_idx >= len(self.mission_goals):
                self.get_logger().info('🏁 ALL mission goals completed!')
                self.current_mode = MODE_STOPPED
                self._publish_motor(0.0, 0.0)
                self._publish_debug()
                return

            # Send next goal to Nav2 immediately
            next_goal = self.mission_goals[self.current_goal_idx]
            self._send_goal_pose(next_goal)
            self.get_logger().info(
                f'→ Advancing to goal {self.current_goal_idx + 1}/'
                f'{len(self.mission_goals)}'
            )
            self._publish_debug()
            return

        # ── MODE DECISION ───────────────────────────────────────────────
        if abs(self.yaw_error) > self.yaw_error_threshold:
            # ═══════════════ NAV2 MODE ═══════════════════════════════════
            new_mode = MODE_NAV2

            if self.current_mode != MODE_NAV2:
                self.get_logger().info(
                    f'🔄 → NAV2 | yaw_err={self.yaw_error:+.3f}rad '
                    f'({math.degrees(self.yaw_error):+.1f}°) > '
                    f'thresh={self.yaw_error_threshold:.2f}'
                )

            # Send goal to Nav2 if not yet sent
            if not self.goal_sent_to_nav2:
                self._send_goal_pose(goal)

            # Bridge /cmd_vel_nav → motors (with auto-retry)
            now = self.get_clock().now()
            dt_cmd = (now - self.last_cmd_vel_time).nanoseconds * 1e-9

            if dt_cmd > self.nav2_timeout:
                # Nav2 stopped → auto-retry
                self._publish_motor(0.0, 0.0)
                self.active_steering = 0.0
                self.active_speed = 0.0

                dt_retry = (now - self.last_retry_time).nanoseconds * 1e-9
                if dt_retry >= self.retry_interval:
                    self.retry_count += 1
                    self.last_retry_time = now
                    self.nav2_status = 'RETRYING'
                    self._send_goal_pose(goal)
                    self.get_logger().warn(
                        f'🔁 RETRY #{self.retry_count}: Nav2 timeout '
                        f'({dt_cmd:.1f}s) → re-sent Goal '
                        f'{self.current_goal_idx + 1}/{len(self.mission_goals)} '
                        f'(dist={dist:.3f}m)'
                    )
                else:
                    self.nav2_status = 'NO_CMD_VEL'
            else:
                # Nav2 active → bridge to motors with MAX steering
                self.nav2_status = 'ACTIVE'

                # Force max steering in Nav2's turn direction
                if abs(self.nav2_angular_z) > 0.01:
                    # Sign of angular.z = turn direction → max angle
                    steering = self.max_angle if self.nav2_angular_z > 0 else -self.max_angle
                else:
                    steering = 0.0

                speed = clamp(
                    self.nav2_speed_scale * self.nav2_linear_x,
                    -self.max_speed, self.max_speed
                )
                self._publish_motor(steering, speed)
                self.active_steering = steering
                self.active_speed = speed

        else:
            # ═══════════════ PID MODE ════════════════════════════════════
            new_mode = MODE_PID

            if self.current_mode != MODE_PID:
                self.get_logger().info(
                    f'🔄 → PID | yaw_err={self.yaw_error:+.3f}rad '
                    f'({math.degrees(self.yaw_error):+.1f}°) <= '
                    f'thresh={self.yaw_error_threshold:.2f}'
                )
                # Reset Nav2 state so it re-sends on next switch
                self.goal_sent_to_nav2 = False
                self.nav2_status = 'IDLE'

            steering = self.lane_steering
            speed = self.lane_speed
            if self.pid_speed_override > 0.0:
                speed = self.pid_speed_override
            self._publish_motor(steering, speed)
            self.active_steering = steering
            self.active_speed = speed

        self.current_mode = new_mode
        self._publish_debug()

    # =====================================================================
    # Send goal to Nav2 via /goal_pose (same as nav2_motor_test)
    # =====================================================================
    def _send_goal_pose(self, goal_msg: PoseStamped):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.map_frame
        pose.pose = goal_msg.pose
        self.goal_pose_pub.publish(pose)
        self.goal_sent_to_nav2 = True
        self.get_logger().info(
            f'📤 /goal_pose: Goal {self.current_goal_idx + 1}/'
            f'{len(self.mission_goals)} → '
            f'({goal_msg.pose.position.x:.2f}, {goal_msg.pose.position.y:.2f})'
        )

    # =====================================================================
    # Motor + debug publishing
    # =====================================================================
    def _publish_motor(self, steering: float, speed: float):
        msg = MotorCommands()
        msg.motor_names = ['steering_angle', 'motor_throttle']
        msg.values = [float(steering), float(speed)]
        self.motor_pub.publish(msg)

    def _publish_debug(self):
        self.pub_mode.publish(Float32(data=float(self.current_mode)))
        self.pub_yaw_error.publish(Float32(data=float(self.yaw_error)))
        self.pub_yaw_robot.publish(Float32(data=float(self.yaw_robot)))
        self.pub_yaw_goal.publish(Float32(data=float(self.yaw_goal)))
        self.pub_yaw_threshold.publish(Float32(data=float(self.yaw_error_threshold)))
        self.pub_nav2_status.publish(String(data=self.nav2_status))
        self.pub_steering.publish(Float32(data=float(self.active_steering)))
        self.pub_speed.publish(Float32(data=float(self.active_speed)))

        total = len(self.mission_goals)
        idx = min(self.current_goal_idx + 1, total)
        self.pub_goal_index.publish(String(data=f'{idx}/{total}'))

        if self.mission_goals and self.current_goal_idx < len(self.mission_goals):
            g = self.mission_goals[self.current_goal_idx]
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = self.map_frame
            pose.pose = g.pose
            self.pub_goal_pose.publish(pose)

    # =====================================================================
    # TF
    # =====================================================================
    def _get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
            return (t.transform.translation.x,
                    t.transform.translation.y,
                    quat_to_yaw(t.transform.rotation))
        except Exception:
            return None


def main(args=None):
    rclpy.init(args=args)
    node = HybridSwitchController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_motor(0.0, 0.0)
        node.get_logger().info('Motors stopped. Shutting down.')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
