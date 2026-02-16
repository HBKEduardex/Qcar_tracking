#!/usr/bin/env python3
"""
Nav2 ↔ Lane Following Bridge Node  (v2 — intersection detection)

Hybrid controller with 3 modes:
  1. LANE_PID   (mode=1.0) — lane visible + Nav2 doesn't want sharp turn → PID steering
  2. NAV2_TURN  (mode=0.0) — Nav2 angular.z exceeds threshold → Nav2 controls steering
                             (intersection detected, robot turns toward goal)
  3. STOPPED    (mode=-1)  — no Nav2 commands or Nav2 says stop

Intersection logic:
  - When |angular.z| > turn_enter_thresh  → enter NAV2_TURN
  - Stay in NAV2_TURN until |angular.z| < turn_exit_thresh  (hysteresis)
  - In NAV2_TURN the robot uses Nav2's angular.z for steering direction
  - Speed is reduced during turns (turn_speed_scale)

Subscribes:
  /cmd_vel_nav           (geometry_msgs/Twist)   — Nav2 velocity command
  /lane/center/error     (std_msgs/Float32)      — lane error [-1, +1]
  /lane/center/visible   (std_msgs/Bool)         — lane detected?
  /lane/curvature        (std_msgs/Float32)      — lane curvature

Publishes:
  /qcar2_motor_speed_cmd (qcar2_interfaces/MotorCommands) — motor commands
  /bridge/steering       (std_msgs/Float32)      — debug: current steering
  /bridge/speed          (std_msgs/Float32)      — debug: current speed
  /bridge/mode           (std_msgs/Float32)      — debug: 1.0=lane, 0.0=nav2_turn, -1.0=stopped
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Twist
from qcar2_interfaces.msg import MotorCommands
from rcl_interfaces.msg import SetParametersResult
import math


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# Modes
MODE_STOPPED   = -1.0
MODE_NAV2_TURN =  0.0
MODE_LANE_PID  =  1.0


class Nav2LaneBridge(Node):
    def __init__(self):
        super().__init__('nav2_lane_bridge')

        # ─── Parameters ───
        # PID for lane steering
        self.declare_parameter('kp', 0.42)
        self.declare_parameter('ki', 0.03)
        self.declare_parameter('kd', 0.17)
        self.declare_parameter('integral_limit', 0.8)
        self.declare_parameter('max_angle', 0.45)

        # Hybrid blending
        self.declare_parameter('nav2_speed_scale', 1.0)
        self.declare_parameter('curve_slowdown_gain', 0.5)
        self.declare_parameter('error_slowdown_gain', 0.3)
        self.declare_parameter('min_speed', 0.05)
        self.declare_parameter('max_speed', 0.30)
        self.declare_parameter('max_steer_rate', 1.5)
        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('lost_timeout', 0.5)

        # ── Intersection turn detection ──
        self.declare_parameter('turn_enter_thresh', 0.4)     # |angular.z| to enter NAV2_TURN (rad/s)
        self.declare_parameter('turn_exit_thresh', 0.15)     # |angular.z| to exit NAV2_TURN (hysteresis)
        self.declare_parameter('turn_steer_gain', 1.0)       # gain on Nav2 angular.z → steering
        self.declare_parameter('turn_speed_scale', 0.5)      # speed multiplier during turns
        self.declare_parameter('turn_max_steer_rate', 3.0)   # faster rate limit during turns

        # Load params
        self.kp = float(self.get_parameter('kp').value)
        self.ki = float(self.get_parameter('ki').value)
        self.kd = float(self.get_parameter('kd').value)
        self.integral_limit = float(self.get_parameter('integral_limit').value)
        self.max_angle = float(self.get_parameter('max_angle').value)
        self.nav2_speed_scale = float(self.get_parameter('nav2_speed_scale').value)
        self.curve_slowdown_gain = float(self.get_parameter('curve_slowdown_gain').value)
        self.error_slowdown_gain = float(self.get_parameter('error_slowdown_gain').value)
        self.min_speed = float(self.get_parameter('min_speed').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.max_steer_rate = float(self.get_parameter('max_steer_rate').value)
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.lost_timeout = float(self.get_parameter('lost_timeout').value)

        self.turn_enter_thresh = float(self.get_parameter('turn_enter_thresh').value)
        self.turn_exit_thresh = float(self.get_parameter('turn_exit_thresh').value)
        self.turn_steer_gain = float(self.get_parameter('turn_steer_gain').value)
        self.turn_speed_scale = float(self.get_parameter('turn_speed_scale').value)
        self.turn_max_steer_rate = float(self.get_parameter('turn_max_steer_rate').value)

        # ─── State ───
        self.lane_error = 0.0
        self.lane_visible = False
        self.lane_curvature = 0.0
        self.nav2_linear_x = 0.0
        self.nav2_angular_z = 0.0
        self.last_nav2_time = self.get_clock().now()

        # PID state
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_steering = 0.0
        self.prev_time = self.get_clock().now()

        # Turn state machine
        self.in_nav2_turn = False  # True = Nav2 controls steering (intersection)

        # ─── Subs ───
        self.sub_cmd_vel = self.create_subscription(
            Twist, '/cmd_vel_nav', self.cb_cmd_vel, 10)
        self.sub_lane_error = self.create_subscription(
            Float32, '/lane/center/error', self.cb_lane_error, 10)
        self.sub_lane_visible = self.create_subscription(
            Bool, '/lane/center/visible', self.cb_lane_visible, 10)
        self.sub_curvature = self.create_subscription(
            Float32, '/lane/curvature', self.cb_curvature, 10)

        # ─── Pub ───
        self.pub_cmd = self.create_publisher(MotorCommands, '/qcar2_motor_speed_cmd', 10)

        # Debug pubs
        self.pub_bridge_steer = self.create_publisher(Float32, '/bridge/steering', 10)
        self.pub_bridge_speed = self.create_publisher(Float32, '/bridge/speed', 10)
        self.pub_bridge_mode = self.create_publisher(Float32, '/bridge/mode', 10)

        # ─── Timer ───
        period = 1.0 / max(1.0, self.rate_hz)
        self.timer = self.create_timer(period, self.control_loop)

        # ─── Param callback ───
        self.add_on_set_parameters_callback(self._on_params_change)

        self.get_logger().info('Nav2LaneBridge v2 started — intersection turn detection enabled')

    # ─── Callbacks ───
    def cb_cmd_vel(self, msg: Twist):
        self.nav2_linear_x = msg.linear.x
        self.nav2_angular_z = msg.angular.z
        self.last_nav2_time = self.get_clock().now()

    def cb_lane_error(self, msg: Float32):
        self.lane_error = msg.data

    def cb_lane_visible(self, msg: Bool):
        self.lane_visible = msg.data

    def cb_curvature(self, msg: Float32):
        self.lane_curvature = msg.data

    # ─── Param change ───
    def _on_params_change(self, params):
        for p in params:
            if hasattr(self, p.name):
                setattr(self, p.name, float(p.value))
        return SetParametersResult(successful=True)

    # ─── Turn state machine ───
    def _update_turn_state(self):
        """Hysteresis-based intersection turn detection."""
        az = abs(self.nav2_angular_z)

        if not self.in_nav2_turn:
            # Enter NAV2_TURN when angular.z is large (intersection)
            if az > self.turn_enter_thresh:
                self.in_nav2_turn = True
                self._reset_pid()
                self.get_logger().info(
                    f'>>> INTERSECTION TURN — Nav2 angular.z={self.nav2_angular_z:+.3f} '
                    f'(thresh={self.turn_enter_thresh:.2f})')
        else:
            # Exit NAV2_TURN when angular.z drops (turn completed)
            if az < self.turn_exit_thresh:
                self.in_nav2_turn = False
                self._reset_pid()
                self.get_logger().info(
                    f'<<< TURN COMPLETE — back to LANE PID '
                    f'(angular.z={self.nav2_angular_z:+.3f})')

    # ─── Main control loop ───
    def control_loop(self):
        now = self.get_clock().now()
        dt_nav2 = (now - self.last_nav2_time).nanoseconds * 1e-9

        # ── STOPPED: no Nav2 data ──
        if dt_nav2 > self.lost_timeout:
            self._publish_motor(0.0, 0.0)
            self._pub_debug(0.0, 0.0, MODE_STOPPED)
            self.in_nav2_turn = False
            return

        # ── STOPPED: Nav2 says stop (goal reached or no goal) ──
        if abs(self.nav2_linear_x) < 0.001:
            self._publish_motor(0.0, 0.0)
            self._pub_debug(0.0, 0.0, MODE_STOPPED)
            self._reset_pid()
            self.in_nav2_turn = False
            return

        # ── Update turn state machine ──
        self._update_turn_state()

        # ── Compute dt ──
        dt = (now - self.prev_time).nanoseconds * 1e-9
        if dt <= 1e-6:
            dt = 1.0 / self.rate_hz
        self.prev_time = now

        # ══════════════════════════════════════════════
        # STEERING decision
        # ══════════════════════════════════════════════
        if self.in_nav2_turn:
            # ── NAV2_TURN: Nav2 controls steering (intersection) ──
            mode = MODE_NAV2_TURN
            # Nav2 angular.z > 0 → turn left, < 0 → turn right
            # Map to steering: positive angular.z → positive steering (left)
            steering_desired = clamp(
                self.turn_steer_gain * self.nav2_angular_z,
                -self.max_angle, self.max_angle
            )
            effective_rate = self.turn_max_steer_rate

        elif self.lane_visible:
            # ── LANE_PID: lane following controls steering ──
            mode = MODE_LANE_PID
            ef = clamp(self.lane_error, -1.0, 1.0)

            derivative = (ef - self.prev_error) / dt
            derivative = clamp(derivative, -8.0, 8.0)
            self.prev_error = ef

            self.integral += ef * dt
            self.integral = clamp(self.integral, -self.integral_limit, self.integral_limit)

            u = self.kp * ef + self.ki * self.integral + self.kd * derivative
            steering_desired = clamp(-u, -self.max_angle, self.max_angle)
            effective_rate = self.max_steer_rate

        else:
            # ── FALLBACK: lane lost + no turn → use Nav2 angular.z gently ──
            mode = MODE_NAV2_TURN
            steering_desired = clamp(
                self.turn_steer_gain * self.nav2_angular_z,
                -self.max_angle, self.max_angle
            )
            effective_rate = self.max_steer_rate
            self._reset_pid()

        # Rate limiter
        max_delta = effective_rate * dt
        delta = clamp(steering_desired - self.prev_steering, -max_delta, max_delta)
        steering = clamp(self.prev_steering + delta, -self.max_angle, self.max_angle)
        self.prev_steering = steering

        # ══════════════════════════════════════════════
        # SPEED decision
        # ══════════════════════════════════════════════
        base_speed = abs(self.nav2_linear_x) * self.nav2_speed_scale

        # Curve/error slowdown
        curv_norm = clamp(abs(self.lane_curvature) * 1000.0, 0.0, 1.0)
        curve_factor = 1.0 - self.curve_slowdown_gain * curv_norm
        curve_factor = clamp(curve_factor, 0.3, 1.0)

        error_factor = 1.0 - self.error_slowdown_gain * abs(self.lane_error)
        error_factor = clamp(error_factor, 0.5, 1.0)

        speed = base_speed * curve_factor * error_factor

        # Extra slowdown during turns
        if self.in_nav2_turn:
            speed *= self.turn_speed_scale

        # Preserve direction
        if self.nav2_linear_x < 0:
            speed = -speed

        speed = clamp(speed, -self.max_speed, self.max_speed)
        if abs(speed) < self.min_speed and abs(self.nav2_linear_x) > 0.001:
            speed = math.copysign(self.min_speed, speed)

        # ── Publish ──
        self._publish_motor(steering, speed)
        self._pub_debug(steering, speed, mode)

    def _publish_motor(self, steering: float, speed: float):
        msg = MotorCommands()
        msg.motor_names = ['motor_throttle', 'steering_angle']
        msg.values = [float(steering), float(speed)]
        self.pub_cmd.publish(msg)

    def _pub_debug(self, steering, speed, mode):
        self.pub_bridge_steer.publish(Float32(data=float(steering)))
        self.pub_bridge_speed.publish(Float32(data=float(speed)))
        self.pub_bridge_mode.publish(Float32(data=float(mode)))

    def _reset_pid(self):
        self.integral = 0.0
        self.prev_error = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = Nav2LaneBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
