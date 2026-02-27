#!/usr/bin/env python3
"""
Mission Lane Bridge — Lane Following + Nav2 for Turns

Modes:
  LANE_PID  (angle < 20°): Lane follower controls → publishes to /qcar2_motor_speed_cmd
  NAV2_TURN (angle > 45°): Sends NavigateToPose to Nav2, forwards /cmd_vel_nav → motors
  STOPPED   (no goal):     Zero commands

Flow:
  /mission_goals → compute angle via TF
    → small angle → LANE_PID (lane follower drives)
    → large angle → NAV2_TURN (Nav2 drives via /cmd_vel_nav)

Subscribes:
  /mission_goals       (PoseStamped)      — semi-goals from planner
  /lane/motor_cmd      (MotorCommands)    — steering+speed from lane follower
  /lane/center/visible (Bool)             — lane detected?
  /lane/curvature      (Float32)          — for speed adaptation
  /cmd_vel_nav         (Twist)            — Nav2 velocity output (used in NAV2_TURN)

Publishes:
  /qcar2_motor_speed_cmd (MotorCommands)  — direct to motors
  /mission_bridge/mode   (Float32)        — debug: 1=LANE, 0=NAV2, -1=STOP
  /mission_bridge/angle  (Float32)        — debug: angle to goal (deg)
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import PoseStamped, Twist
from qcar2_interfaces.msg import MotorCommands
from nav2_msgs.action import NavigateToPose

import tf2_ros


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


MODE_STOPPED   = -1.0
MODE_NAV2_TURN =  0.0
MODE_LANE_PID  =  1.0


class MissionLaneBridge(Node):
    def __init__(self):
        super().__init__('mission_lane_bridge')

        # ─── Parameters ───
        self.declare_parameter('turn_enter_angle_deg', 45.0)
        self.declare_parameter('turn_exit_angle_deg', 20.0)
        self.declare_parameter('base_speed', 0.25)
        self.declare_parameter('max_speed', 0.30)
        self.declare_parameter('min_speed', 0.05)
        self.declare_parameter('max_angle', 0.6)
        self.declare_parameter('curve_slowdown_gain', 0.4)
        self.declare_parameter('goal_reached_dist', 0.5)
        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('goal_timeout', 5.0)
        self.declare_parameter('lane_timeout', 0.5)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')

        self.turn_enter = math.radians(
            float(self.get_parameter('turn_enter_angle_deg').value))
        self.turn_exit = math.radians(
            float(self.get_parameter('turn_exit_angle_deg').value))
        self.base_speed = float(self.get_parameter('base_speed').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.min_speed = float(self.get_parameter('min_speed').value)
        self.max_angle = float(self.get_parameter('max_angle').value)
        self.curve_gain = float(self.get_parameter('curve_slowdown_gain').value)
        self.goal_dist = float(self.get_parameter('goal_reached_dist').value)
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.goal_timeout = float(self.get_parameter('goal_timeout').value)
        self.lane_timeout = float(self.get_parameter('lane_timeout').value)
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        # ─── State ───
        self.lane_steering = 0.0
        self.lane_speed = 0.0
        self.lane_visible = False
        self.lane_curvature = 0.0
        self.last_lane_time = self.get_clock().now()

        self.goal_x = None
        self.goal_y = None
        self.last_goal_time = self.get_clock().now()
        self.has_goal = False

        self.in_nav2_turn = False
        self.nav2_steering = 0.0
        self.nav2_speed = 0.0
        self.nav2_goal_handle = None
        self.nav2_active = False

        # ─── TF ───
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ─── Nav2 action client ───
        self.nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose')

        # ─── Subs ───
        self.create_subscription(
            PoseStamped, '/mission_goals', self.cb_mission_goal, 10)
        self.create_subscription(
            MotorCommands, '/lane/motor_cmd', self.cb_lane_cmd, 10)
        self.create_subscription(
            Bool, '/lane/center/visible', self.cb_lane_visible, 10)
        self.create_subscription(
            Float32, '/lane/curvature', self.cb_curvature, 10)
        self.create_subscription(
            Twist, '/cmd_vel_nav', self.cb_cmd_vel_nav, 10)

        # ─── Pubs ───
        self.pub_cmd = self.create_publisher(
            MotorCommands, '/qcar2_motor_speed_cmd', 10)
        self.pub_mode = self.create_publisher(
            Float32, '/mission_bridge/mode', 10)
        self.pub_angle = self.create_publisher(
            Float32, '/mission_bridge/angle', 10)

        # ─── Timer ───
        period = 1.0 / max(1.0, self.rate_hz)
        self.create_timer(period, self.control_loop)

        self.get_logger().info(
            f'MissionLaneBridge started\n'
            f'  LANE_PID:  angle < {math.degrees(self.turn_exit):.0f}°\n'
            f'  NAV2_TURN: angle > {math.degrees(self.turn_enter):.0f}°\n'
            f'  Output:    /qcar2_motor_speed_cmd (direct to motors)')

    # ─── Callbacks ───
    def cb_mission_goal(self, msg: PoseStamped):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.last_goal_time = self.get_clock().now()
        self.has_goal = True
        self.get_logger().info(
            f'📍 Semi-goal: ({self.goal_x:.2f}, {self.goal_y:.2f})')

    def cb_lane_cmd(self, msg: MotorCommands):
        for i, name in enumerate(msg.motor_names):
            if name == 'steering_angle' and i < len(msg.values):
                self.lane_steering = msg.values[i]
            elif name == 'motor_throttle' and i < len(msg.values):
                self.lane_speed = msg.values[i]
        self.last_lane_time = self.get_clock().now()

    def cb_lane_visible(self, msg: Bool):
        self.lane_visible = msg.data

    def cb_curvature(self, msg: Float32):
        self.lane_curvature = msg.data

    def cb_cmd_vel_nav(self, msg: Twist):
        """Nav2 velocity → store for forwarding in NAV2_TURN mode."""
        self.nav2_steering = msg.angular.z
        self.nav2_speed = msg.linear.x

    # ─── TF ───
    def _get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame,
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1))
            x = t.transform.translation.x
            y = t.transform.translation.y
            yaw = 2.0 * math.atan2(t.transform.rotation.z, t.transform.rotation.w)
            return (x, y, yaw)
        except Exception:
            return None

    def _angle_to_goal(self, rx, ry, ryaw, gx, gy):
        direction = math.atan2(gy - ry, gx - rx)
        angle = math.atan2(math.sin(direction - ryaw), math.cos(direction - ryaw))
        return angle

    # ─── Nav2 ───
    def _send_nav2_goal(self, gx, gy, yaw=0.0):
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('Nav2 not available for turn!')
            return
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = self.map_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(gx)
        goal.pose.pose.position.y = float(gy)
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(f'🔄 NAV2_TURN → ({gx:.2f}, {gy:.2f})')
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self._nav2_response_cb)

    def _nav2_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('Nav2 rejected turn goal!')
            self.nav2_active = False
            return
        self.nav2_goal_handle = handle
        self.nav2_active = True
        result_future = handle.get_result_async()
        result_future.add_done_callback(self._nav2_result_cb)

    def _nav2_result_cb(self, future):
        status = future.result().status
        if status == 4:
            self.get_logger().info('★ Nav2 turn complete → back to LANE_PID')
        elif status == 5:
            self.get_logger().info('Nav2 turn cancelled')
        else:
            self.get_logger().warn(f'Nav2 turn ended status={status}')
        self.nav2_active = False
        self.in_nav2_turn = False

    def _cancel_nav2(self):
        if self.nav2_active and self.nav2_goal_handle is not None:
            self.nav2_goal_handle.cancel_goal_async()
            self.nav2_active = False

    # ─── Main loop ───
    def control_loop(self):
        now = self.get_clock().now()
        dt_goal = (now - self.last_goal_time).nanoseconds * 1e-9
        dt_lane = (now - self.last_lane_time).nanoseconds * 1e-9

        # No goal → lane passthrough or stop
        if not self.has_goal or dt_goal > self.goal_timeout:
            if dt_lane < self.lane_timeout and self.lane_visible:
                self._pub_lane()
                self._debug(MODE_LANE_PID, 0.0)
            else:
                self._pub_motor(0.0, 0.0)
                self._debug(MODE_STOPPED, 0.0)
            return

        # Get pose
        pose = self._get_robot_pose()
        if pose is None:
            if dt_lane < self.lane_timeout and self.lane_visible:
                self._pub_lane()
                self._debug(MODE_LANE_PID, 0.0)
            else:
                self._pub_motor(0.0, 0.0)
                self._debug(MODE_STOPPED, 0.0)
            return

        rx, ry, ryaw = pose
        dist = math.hypot(self.goal_x - rx, self.goal_y - ry)
        angle = self._angle_to_goal(rx, ry, ryaw, self.goal_x, self.goal_y)
        angle_abs = abs(angle)

        # Goal reached → lane passthrough
        if dist < self.goal_dist:
            self._cancel_nav2()
            self.in_nav2_turn = False
            if dt_lane < self.lane_timeout and self.lane_visible:
                self._pub_lane()
                self._debug(MODE_LANE_PID, 0.0)
            else:
                self._pub_motor(0.0, 0.0)
                self._debug(MODE_STOPPED, 0.0)
            return

        # ── Hysteresis mode switch ──
        if not self.in_nav2_turn and angle_abs > self.turn_enter:
            self.in_nav2_turn = True
            self.get_logger().info(
                f'>>> NAV2_TURN — angle={math.degrees(angle_abs):.0f}° '
                f'> {math.degrees(self.turn_enter):.0f}°')
            self._send_nav2_goal(self.goal_x, self.goal_y, angle)

        elif self.in_nav2_turn and angle_abs < self.turn_exit:
            self.in_nav2_turn = False
            self._cancel_nav2()
            self.get_logger().info(
                f'<<< LANE_PID — angle={math.degrees(angle_abs):.0f}° '
                f'< {math.degrees(self.turn_exit):.0f}°')

        # ── Publish based on mode ──
        if self.in_nav2_turn:
            # NAV2 drives: forward /cmd_vel_nav as MotorCommands
            self._pub_motor(self.nav2_steering, self.nav2_speed)
            self._debug(MODE_NAV2_TURN, math.degrees(angle))
        elif dt_lane < self.lane_timeout and self.lane_visible:
            # Lane follower drives
            self._pub_lane()
            self._debug(MODE_LANE_PID, math.degrees(angle))
        else:
            # No lane, small angle → stop and wait
            self._pub_motor(0.0, 0.0)
            self._debug(MODE_STOPPED, math.degrees(angle))

    # ─── Publish helpers ───
    def _pub_lane(self):
        """Publish lane follower commands with curve slowdown."""
        steering = clamp(self.lane_steering, -self.max_angle, self.max_angle)
        speed = abs(self.lane_speed) if abs(self.lane_speed) > 0.001 else self.base_speed
        curv = clamp(abs(self.lane_curvature) * 1000.0, 0.0, 1.0)
        speed *= clamp(1.0 - self.curve_gain * curv, 0.3, 1.0)
        speed = clamp(speed, self.min_speed, self.max_speed)
        self._pub_motor(steering, speed)

    def _pub_motor(self, steering, speed):
        msg = MotorCommands()
        msg.motor_names = ['steering_angle', 'motor_throttle']
        msg.values = [float(steering), float(speed)]
        self.pub_cmd.publish(msg)

    def _debug(self, mode, angle_deg):
        self.pub_mode.publish(Float32(data=float(mode)))
        self.pub_angle.publish(Float32(data=float(angle_deg)))


def main(args=None):
    rclpy.init(args=args)
    node = MissionLaneBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
