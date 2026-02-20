#!/usr/bin/env python3
"""
Hybrid Nav2 ↔ Lane Following Controller (v3 — MotorCommands interface)

Architecture:
  - Receives MotorCommands from yellow_line_follower_controller via /lane/motor_cmd
    (steering_angle in rad, motor_throttle in m/s — normalizado per qcar2_interfaces)
  - Receives cmd_vel from Nav2 via /cmd_vel_nav
  - Detects intersections (high angular.z) and switches modes
  - Publishes MotorCommands to /qcar2_motor_speed_cmd

Modes:
  1. LANE_PID   (mode=1.0) — steering+speed from lane follower MotorCommands
  2. NAV2_TURN  (mode=0.0) — steering from Nav2 angular.z (intersection) + reduced speed
  3. STOPPED    (mode=-1)  — no commands or Nav2 says stop

Subscribes:
  /cmd_vel_nav       (geometry_msgs/Twist)            — Nav2 velocity command
  /lane/motor_cmd    (qcar2_interfaces/MotorCommands)  — steering+speed from lane follower
  /lane/center/visible (std_msgs/Bool)                — lane detected?
  /lane/curvature    (std_msgs/Float32)               — lane curvature (optional)

Publishes:
  /qcar2_motor_speed_cmd (qcar2_interfaces/MotorCommands) — final motor commands
  /hybrid/steering       (std_msgs/Float32)      — debug: current steering (rad)
  /hybrid/speed          (std_msgs/Float32)      — debug: current speed (m/s)
  /hybrid/mode           (std_msgs/Float32)      — debug: 1.0=lane, 0.0=nav2_turn, -1.0=stopped
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


class HybridController(Node):
    def __init__(self):
        super().__init__('hybrid_controller')

        # ─── Parameters ───
        # Speed control
        self.declare_parameter('nav2_speed_scale', 1.0)
        self.declare_parameter('base_autonomous_speed', 0.25)  # if no Nav2 speed
        self.declare_parameter('curve_slowdown_gain', 0.5)
        self.declare_parameter('error_slowdown_gain', 0.3)
        self.declare_parameter('min_speed', 0.05)
        self.declare_parameter('max_speed', 0.30)
        self.declare_parameter('max_steer_rate', 1.5)
        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('lost_timeout', 0.5)
        self.declare_parameter('max_angle', 0.45)            # max steering angle (rad)
        self.declare_parameter('lane_cmd_topic', '/lane/motor_cmd')  # MotorCommands from lane follower

        # ── Intersection turn detection ──
        self.declare_parameter('turn_enter_thresh', 0.4)     # |angular.z| to enter NAV2_TURN (rad/s)
        self.declare_parameter('turn_exit_thresh', 0.15)     # |angular.z| to exit NAV2_TURN (hysteresis)
        self.declare_parameter('turn_steer_gain', 1.0)       # gain on Nav2 angular.z → steering
        self.declare_parameter('turn_speed_scale', 0.5)      # speed multiplier during turns
        self.declare_parameter('turn_max_steer_rate', 3.0)   # faster rate limit during turns

        # Load params
        self.nav2_speed_scale = float(self.get_parameter('nav2_speed_scale').value)
        self.base_autonomous_speed = float(self.get_parameter('base_autonomous_speed').value)
        self.curve_slowdown_gain = float(self.get_parameter('curve_slowdown_gain').value)
        self.error_slowdown_gain = float(self.get_parameter('error_slowdown_gain').value)
        self.min_speed = float(self.get_parameter('min_speed').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.max_steer_rate = float(self.get_parameter('max_steer_rate').value)
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.lost_timeout = float(self.get_parameter('lost_timeout').value)
        self.max_angle = float(self.get_parameter('max_angle').value)
        self.lane_cmd_topic = self.get_parameter('lane_cmd_topic').value

        self.turn_enter_thresh = float(self.get_parameter('turn_enter_thresh').value)
        self.turn_exit_thresh = float(self.get_parameter('turn_exit_thresh').value)
        self.turn_steer_gain = float(self.get_parameter('turn_steer_gain').value)
        self.turn_speed_scale = float(self.get_parameter('turn_speed_scale').value)
        self.turn_max_steer_rate = float(self.get_parameter('turn_max_steer_rate').value)

        # ─── State ───
        self.lane_steering = 0.0          # from lane follower MotorCommands (rad)
        self.lane_speed = 0.0             # from lane follower MotorCommands (m/s)
        self.prev_steering = 0.0          # for rate limiter
        self.lane_visible = False
        self.lane_curvature = 0.0
        self.nav2_linear_x = 0.0
        self.nav2_angular_z = 0.0
        self.last_nav2_time = self.get_clock().now()
        self.last_steering_time = self.get_clock().now()

        # Turn state machine
        self.in_nav2_turn = False

        # ─── Subs ───
        self.sub_cmd_vel = self.create_subscription(
            Twist, '/cmd_vel_nav', self.cb_cmd_vel, 10)
        self.sub_lane_cmd = self.create_subscription(
            MotorCommands, self.lane_cmd_topic, self.cb_lane_cmd, 10)
        self.sub_lane_visible = self.create_subscription(
            Bool, '/lane/center/visible', self.cb_lane_visible, 10)
        self.sub_curvature = self.create_subscription(
            Float32, '/lane/curvature', self.cb_curvature, 10)

        # ─── Pub ───
        self.pub_cmd = self.create_publisher(MotorCommands, '/qcar2_motor_speed_cmd', 10)

        # Debug pubs
        self.pub_hybrid_steer = self.create_publisher(Float32, '/hybrid/steering', 10)
        self.pub_hybrid_speed = self.create_publisher(Float32, '/hybrid/speed', 10)
        self.pub_hybrid_mode = self.create_publisher(Float32, '/hybrid/mode', 10)

        # ─── Timer ───
        period = 1.0 / max(1.0, self.rate_hz)
        self.timer = self.create_timer(period, self.control_loop)

        # ─── Param callback ───
        self.add_on_set_parameters_callback(self._on_params_change)

        self.get_logger().info('HybridController v3 started (MotorCommands interface)')
        self.get_logger().info(f'  Subscribes to {self.lane_cmd_topic} (MotorCommands from lane follower)')
        self.get_logger().info(f'  Subscribes to /cmd_vel_nav (Nav2 velocity)')
        self.get_logger().info(f'  max_angle={self.max_angle} rad')
        self.get_logger().info(f'  base_autonomous_speed={self.base_autonomous_speed} (fallback)')
        self.get_logger().info(f'  Publishes to /qcar2_motor_speed_cmd (MotorCommands)')

    # ─── Callbacks ───
    def cb_cmd_vel(self, msg: Twist):
        self.nav2_linear_x = msg.linear.x
        self.nav2_angular_z = msg.angular.z
        self.last_nav2_time = self.get_clock().now()

    def cb_lane_cmd(self, msg: MotorCommands):
        """Parse MotorCommands normalizado: steering_angle (rad), motor_throttle (m/s)."""
        for i, name in enumerate(msg.motor_names):
            if name == 'steering_angle' and i < len(msg.values):
                self.lane_steering = msg.values[i]
            elif name == 'motor_throttle' and i < len(msg.values):
                self.lane_speed = msg.values[i]
        self.last_steering_time = self.get_clock().now()

    def cb_lane_visible(self, msg: Bool):
        self.lane_visible = msg.data

    def cb_curvature(self, msg: Float32):
        self.lane_curvature = msg.data

    # ─── Param change ───
    def _on_params_change(self, params):
        for p in params:
            if hasattr(self, p.name):
                try:
                    setattr(self, p.name, float(p.value) if p.name != 'rate_hz' else float(p.value))
                except (ValueError, TypeError):
                    pass
        return SetParametersResult(successful=True)

    # ─── Turn state machine ───
    def _update_turn_state(self):
        """Hysteresis-based intersection turn detection."""
        az = abs(self.nav2_angular_z)

        if not self.in_nav2_turn:
            # Enter NAV2_TURN when angular.z is large (intersection)
            if az > self.turn_enter_thresh:
                self.in_nav2_turn = True
                self.get_logger().info(
                    f'>>> INTERSECTION DETECTED — Nav2 angular.z={self.nav2_angular_z:+.3f} '
                    f'(enter_thresh={self.turn_enter_thresh:.2f})')
        else:
            # Exit NAV2_TURN when angular.z drops (turn completed)
            if az < self.turn_exit_thresh:
                self.in_nav2_turn = False
                self.get_logger().info(
                    f'<<< TURN COMPLETE — back to LANE steering '
                    f'(angular.z={self.nav2_angular_z:+.3f})')

    # ─── Main control loop ───
    def control_loop(self):
        now = self.get_clock().now()
        dt_nav2 = (now - self.last_nav2_time).nanoseconds * 1e-9
        dt_steering = (now - self.last_steering_time).nanoseconds * 1e-9

        # ── STOPPED: no Nav2 data for too long ──
        if dt_nav2 > self.lost_timeout:
            self._publish_motor(0.0, 0.0)
            self._pub_debug(0.0, 0.0, MODE_STOPPED)
            self.in_nav2_turn = False
            self.prev_steering = 0.0
            return

        # ── STOPPED: Nav2 says stop (goal reached or no goal) ──
        if abs(self.nav2_linear_x) < 0.001:
            self._publish_motor(0.0, 0.0)
            self._pub_debug(0.0, 0.0, MODE_STOPPED)
            self.in_nav2_turn = False
            self.prev_steering = 0.0
            return

        # ── Update turn state machine ──
        self._update_turn_state()

        # ══════════════════════════════════════════════
        # STEERING decision
        # ══════════════════════════════════════════════
        if self.in_nav2_turn:
            # ── NAV2_TURN: Nav2 controls steering (intersection) ──
            mode = MODE_NAV2_TURN
            steering_desired = clamp(
                self.turn_steer_gain * self.nav2_angular_z,
                -self.max_angle, self.max_angle
            )
            effective_rate = self.turn_max_steer_rate

        elif dt_steering < self.lost_timeout and self.lane_visible:
            # ── LANE_PID: steering from lane follower MotorCommands (already in rad) ──
            mode = MODE_LANE_PID
            steering_desired = clamp(self.lane_steering, -self.max_angle, self.max_angle)
            effective_rate = self.max_steer_rate

        else:
            # ── FALLBACK: lane lost + no turn → use Nav2 angular.z gently ──
            mode = MODE_NAV2_TURN
            steering_desired = clamp(
                self.turn_steer_gain * self.nav2_angular_z,
                -self.max_angle, self.max_angle
            )
            effective_rate = self.max_steer_rate

        # Rate limiter on steering (smooth transitions)
        dt = 1.0 / max(1.0, self.rate_hz)
        max_delta = effective_rate * dt
        delta = clamp(steering_desired - self.prev_steering, -max_delta, max_delta)
        steering = clamp(self.prev_steering + delta, -self.max_angle, self.max_angle)
        self.prev_steering = steering

        # ══════════════════════════════════════════════
        # SPEED decision (prioridad: Nav2 > lane_follower > base_autonomous)
        # ════════════════════════════════════════════
        # Prioridad 1: Nav2 proporciona velocidad
        if abs(self.nav2_linear_x) > 0.001:
            base_speed = abs(self.nav2_linear_x) * self.nav2_speed_scale
        # Prioridad 2: lane_follower envió velocidad en MotorCommands (send_speed_in_motor_cmd=true)
        elif abs(self.lane_speed) > 0.001 and self.lane_visible:
            base_speed = abs(self.lane_speed)
        # Prioridad 3: usar velocidad autónoma base si lane es visible
        elif self.lane_visible:
            base_speed = self.base_autonomous_speed
        # Sin lane visible y sin Nav2 → parar
        else:
            base_speed = 0.0

        # Curve/error slowdown (even if using Nav2 speed)
        curv_norm = clamp(abs(self.lane_curvature) * 1000.0, 0.0, 1.0)
        curve_factor = 1.0 - self.curve_slowdown_gain * curv_norm
        curve_factor = clamp(curve_factor, 0.3, 1.0)
        
        speed = base_speed * curve_factor

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
        """
        Publish MotorCommands with correct units:
        - steering_angle: radians (rad)
        - motor_throttle: m/s
        """
        msg = MotorCommands()
        msg.motor_names = ['steering_angle', 'motor_throttle']
        msg.values = [float(steering), float(speed)]  # steering in rad, speed in m/s
        self.pub_cmd.publish(msg)

    def _pub_debug(self, steering, speed, mode):
        self.pub_hybrid_steer.publish(Float32(data=float(steering)))
        self.pub_hybrid_speed.publish(Float32(data=float(speed)))
        self.pub_hybrid_mode.publish(Float32(data=float(mode)))


def main(args=None):
    rclpy.init(args=args)
    node = HybridController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
