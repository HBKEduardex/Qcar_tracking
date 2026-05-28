#!/usr/bin/env python3
"""
MPPI Tracking Controller for QCar2
===================================
Custom Model Predictive Path Integral controller that replaces Nav2
for continuous semigoal trajectory following.

Subscribes:
  /mission_goals  (geometry_msgs/PoseStamped)  — semigoals from planner

Publishes:
  /nav2/motor_cmd (qcar2_interfaces/MotorCommands) — through mixer safety chain
  /mppi_tracking/debug (std_msgs/String) — optional debug info

Pose:
  TF2 lookup: pgm_map → base_link

Author: QCar2 Team — ACC 2026
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Float32
from qcar2_interfaces.msg import MotorCommands
import tf2_ros


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def normalize_angle(a):
    """Normalize angle to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def quat_to_yaw(q):
    """Convert quaternion to yaw using full atan2 formula."""
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y ** 2 + q.z ** 2),
    )


class MPPITrackingNode(Node):
    """MPPI controller for continuous semigoal trajectory tracking."""

    def __init__(self):
        super().__init__('mppi_tracking_node')

        # ── Declare all parameters ──────────────────────────────────────
        self._declare_params()
        self._load_params()

        # ── State ───────────────────────────────────────────────────────
        self.semigoals = []           # list of (x, y, yaw)
        self.current_sg_idx = 0      # index into semigoals
        self.prev_speed = 0.0        # previous command for smoothness
        self.prev_steering = 0.0     # previous command for smoothness
        self.last_pose_time = None
        self.last_sg_time = None

        # ── TF2 ─────────────────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Subscriber: semigoals from planner ──────────────────────────
        self.create_subscription(
            PoseStamped, self.semigoals_topic,
            self._semigoal_cb, 10)

        # ── Publisher: motor commands to mixer ──────────────────────────
        self.pub_cmd = self.create_publisher(
            MotorCommands, self.cmd_output_topic, 10)

        # ── Debug publishers ────────────────────────────────────────────
        if self.publish_debug:
            self.pub_debug = self.create_publisher(
                String, self.debug_topic, 10)
            self.pub_speed_dbg = self.create_publisher(
                Float32, '/mppi_tracking/speed', 10)
            self.pub_steer_dbg = self.create_publisher(
                Float32, '/mppi_tracking/steering', 10)
            self.pub_sg_idx_dbg = self.create_publisher(
                Float32, '/mppi_tracking/sg_index', 10)

        # ── Control timer ───────────────────────────────────────────────
        period = 1.0 / max(1.0, self.control_frequency)
        self.create_timer(period, self._control_loop)

        self.get_logger().info(
            f'MPPITrackingNode started\n'
            f'  semigoals: {self.semigoals_topic}\n'
            f'  output:    {self.cmd_output_topic}\n'
            f'  frame:     {self.map_frame} → {self.base_frame}\n'
            f'  frequency: {self.control_frequency} Hz\n'
            f'  samples:   {self.num_samples}, horizon: {self.horizon_steps}')

    # ═════════════════════════════════════════════════════════════════════
    # Parameters
    # ═════════════════════════════════════════════════════════════════════
    def _declare_params(self):
        d = self.declare_parameter
        d('enabled', True)
        d('map_frame', 'pgm_map')
        d('base_frame', 'base_link')
        d('semigoals_topic', '/mission_goals')
        d('cmd_output_topic', '/nav2/motor_cmd')
        d('control_frequency', 20.0)
        # MPPI core
        d('horizon_steps', 15)
        d('dt', 0.05)
        d('num_samples', 500)
        d('lambda_temperature', 1.0)
        # Velocity limits
        d('max_speed', 0.45)
        d('min_speed', 0.05)
        d('max_reverse_speed', 0.0)
        d('max_steering_angle', 0.65)
        d('wheelbase', 0.256)
        # Noise
        d('speed_noise_std', 0.12)
        d('steering_noise_std', 0.25)
        # Cost weights
        d('weight_position_error', 8.0)
        d('weight_yaw_error', 2.5)
        d('weight_path_tracking', 5.0)
        d('weight_control_effort', 0.2)
        d('weight_control_smoothness', 0.8)
        d('weight_progress', 2.0)
        d('weight_terminal_error', 10.0)
        # Semigoal management
        d('lookahead_semigoals', 5)
        d('semigoal_reached_distance', 0.12)
        d('semigoal_reached_yaw_deg', 25.0)
        d('allow_skip_semigoals', True)
        d('max_semigoal_skip', 3)
        # Safety
        d('stop_if_no_pose', True)
        d('stop_if_no_semigoals', True)
        d('command_timeout_sec', 0.5)
        # Debug
        d('publish_debug', True)
        d('debug_topic', '/mppi_tracking/debug')

    def _load_params(self):
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.enabled = bool(g('enabled'))
        self.map_frame = str(g('map_frame'))
        self.base_frame = str(g('base_frame'))
        self.semigoals_topic = str(g('semigoals_topic'))
        self.cmd_output_topic = str(g('cmd_output_topic'))
        self.control_frequency = float(g('control_frequency'))
        self.horizon_steps = int(g('horizon_steps'))
        self.dt = float(g('dt'))
        self.num_samples = int(g('num_samples'))
        self.lambda_temp = float(g('lambda_temperature'))
        self.max_speed = float(g('max_speed'))
        self.min_speed = float(g('min_speed'))
        self.max_reverse_speed = float(g('max_reverse_speed'))
        self.max_steering_angle = float(g('max_steering_angle'))
        self.wheelbase = float(g('wheelbase'))
        self.speed_noise_std = float(g('speed_noise_std'))
        self.steering_noise_std = float(g('steering_noise_std'))
        self.w_pos = float(g('weight_position_error'))
        self.w_yaw = float(g('weight_yaw_error'))
        self.w_path = float(g('weight_path_tracking'))
        self.w_effort = float(g('weight_control_effort'))
        self.w_smooth = float(g('weight_control_smoothness'))
        self.w_progress = float(g('weight_progress'))
        self.w_terminal = float(g('weight_terminal_error'))
        self.lookahead = int(g('lookahead_semigoals'))
        self.sg_reach_dist = float(g('semigoal_reached_distance'))
        self.sg_reach_yaw = math.radians(float(g('semigoal_reached_yaw_deg')))
        self.allow_skip = bool(g('allow_skip_semigoals'))
        self.max_skip = int(g('max_semigoal_skip'))
        self.stop_no_pose = bool(g('stop_if_no_pose'))
        self.stop_no_sg = bool(g('stop_if_no_semigoals'))
        self.cmd_timeout = float(g('command_timeout_sec'))
        self.publish_debug = bool(g('publish_debug'))
        self.debug_topic = str(g('debug_topic'))

    # ═════════════════════════════════════════════════════════════════════
    # Callbacks
    # ═════════════════════════════════════════════════════════════════════
    def _semigoal_cb(self, msg: PoseStamped):
        """Receive semigoal from planner, accumulate into trajectory buffer."""
        x = msg.pose.position.x
        y = msg.pose.position.y
        yaw = quat_to_yaw(msg.pose.orientation)

        # Detect new path: if this semigoal is far from the last one
        # or the buffer is empty, it's likely a new trajectory
        if self.semigoals:
            lx, ly, _ = self.semigoals[-1]
            d = math.hypot(x - lx, y - ly)
            # If new semigoal is behind current progress or very far, reset
            if d > 3.0:
                self.get_logger().info(
                    f'🔄 New path detected (d={d:.2f}m) — resetting buffer')
                self.semigoals.clear()
                self.current_sg_idx = 0

        # Avoid duplicate semigoals
        if self.semigoals:
            lx, ly, _ = self.semigoals[-1]
            if math.hypot(x - lx, y - ly) < 0.03:
                return

        self.semigoals.append((x, y, yaw))
        self.last_sg_time = self.get_clock().now()

        self.get_logger().info(
            f'📍 Semigoal #{len(self.semigoals)}: '
            f'({x:.2f}, {y:.2f}, yaw={math.degrees(yaw):.0f}°)',
            throttle_duration_sec=0.5)

    # ═════════════════════════════════════════════════════════════════════
    # Robot pose from TF2
    # ═════════════════════════════════════════════════════════════════════
    def _get_robot_pose(self):
        """Get robot (x, y, yaw) from TF2. Returns None on failure."""
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
            x = t.transform.translation.x
            y = t.transform.translation.y
            yaw = quat_to_yaw(t.transform.rotation)
            self.last_pose_time = self.get_clock().now()
            return x, y, yaw
        except Exception:
            return None

    # ═════════════════════════════════════════════════════════════════════
    # Semigoal management
    # ═════════════════════════════════════════════════════════════════════
    def _advance_semigoals(self, rx, ry, ryaw):
        """Advance current semigoal index based on proximity and progress."""
        if not self.semigoals or self.current_sg_idx >= len(self.semigoals):
            return

        # Check if current semigoal is reached
        sx, sy, syaw = self.semigoals[self.current_sg_idx]
        dist = math.hypot(sx - rx, sy - ry)

        if dist <= self.sg_reach_dist:
            self.current_sg_idx += 1
            if self.current_sg_idx < len(self.semigoals):
                nx, ny, _ = self.semigoals[self.current_sg_idx]
                self.get_logger().info(
                    f'✓ Semigoal {self.current_sg_idx}/{len(self.semigoals)} '
                    f'reached → next ({nx:.2f}, {ny:.2f})')
            return

        # Skip semigoals the robot has already passed
        if self.allow_skip:
            best_idx = self.current_sg_idx
            best_dot = -2.0
            limit = min(self.current_sg_idx + self.max_skip + 1,
                        len(self.semigoals))
            for i in range(self.current_sg_idx, limit):
                gx, gy, _ = self.semigoals[i]
                dx, dy = gx - rx, gy - ry
                d = math.hypot(dx, dy)
                if d < 0.01:
                    # Robot is on top of this semigoal
                    best_idx = i + 1
                    continue
                # Check if semigoal is ahead of robot (dot product with heading)
                fwd_x = math.cos(ryaw)
                fwd_y = math.sin(ryaw)
                dot = (dx * fwd_x + dy * fwd_y) / d
                if dot > best_dot and d < self.sg_reach_dist * 3.0:
                    best_dot = dot
                    best_idx = i

            if best_idx > self.current_sg_idx:
                self.current_sg_idx = best_idx
                if self.current_sg_idx < len(self.semigoals):
                    self.get_logger().info(
                        f'⏭ Skipped to semigoal {self.current_sg_idx}/'
                        f'{len(self.semigoals)}')

    def _get_reference_points(self):
        """Get reference semigoals for the MPPI horizon."""
        if not self.semigoals or self.current_sg_idx >= len(self.semigoals):
            return []
        end = min(self.current_sg_idx + self.lookahead, len(self.semigoals))
        return self.semigoals[self.current_sg_idx:end]

    # ═════════════════════════════════════════════════════════════════════
    # MPPI Core
    # ═════════════════════════════════════════════════════════════════════
    def _run_mppi(self, rx, ry, ryaw, ref_points):
        """
        Run MPPI optimization.
        Returns (best_speed, best_steering).
        """
        N = self.num_samples
        H = self.horizon_steps
        dt = self.dt

        # Generate noise samples for speed and steering
        speed_noise = np.random.normal(0.0, self.speed_noise_std, (N, H))
        steer_noise = np.random.normal(0.0, self.steering_noise_std, (N, H))

        # Base control: drive toward first reference point
        if ref_points:
            tx, ty, tyaw = ref_points[0]
            angle_to_target = math.atan2(ty - ry, tx - rx)
            angle_err = normalize_angle(angle_to_target - ryaw)
            base_steer = clamp(angle_err * 1.5, -self.max_steering_angle,
                               self.max_steering_angle)
            base_speed = self.max_speed * 0.7
        else:
            base_steer = 0.0
            base_speed = 0.0

        # Build control sequences: base + noise
        speeds = base_speed + speed_noise      # (N, H)
        steers = base_steer + steer_noise      # (N, H)

        # Clamp controls
        speeds = np.clip(speeds, -self.max_reverse_speed, self.max_speed)
        steers = np.clip(steers, -self.max_steering_angle,
                         self.max_steering_angle)

        # Ensure minimum speed when moving forward
        fwd_mask = speeds > 0.0
        speeds[fwd_mask] = np.maximum(speeds[fwd_mask], self.min_speed)

        # ── Forward simulate all trajectories (Ackermann kinematics) ────
        # State: (x, y, yaw) for each sample
        x = np.full(N, rx)
        y = np.full(N, ry)
        yaw = np.full(N, ryaw)

        costs = np.zeros(N)
        ref_arr = np.array(ref_points)  # (R, 3): x, y, yaw

        for t_step in range(H):
            v = speeds[:, t_step]
            s = steers[:, t_step]

            # Ackermann kinematics: angular velocity from steering
            omega = v * np.tan(s) / self.wheelbase

            # Update state
            x += v * np.cos(yaw) * dt
            y += v * np.sin(yaw) * dt
            yaw += omega * dt

            # ── Cost: position error to nearest reference point ─────────
            if len(ref_arr) > 0:
                # Distance to each reference point, take minimum
                dx = x[:, None] - ref_arr[None, :, 0]   # (N, R)
                dy = y[:, None] - ref_arr[None, :, 1]    # (N, R)
                dists = np.sqrt(dx ** 2 + dy ** 2)       # (N, R)
                min_dist = np.min(dists, axis=1)          # (N,)
                costs += self.w_path * min_dist

                # Distance to current target (first ref point)
                d_target = np.sqrt(
                    (x - ref_arr[0, 0]) ** 2 +
                    (y - ref_arr[0, 1]) ** 2)
                costs += self.w_pos * d_target

                # Yaw error to current target
                yaw_err = np.abs(np.arctan2(
                    np.sin(yaw - ref_arr[0, 2]),
                    np.cos(yaw - ref_arr[0, 2])))
                costs += self.w_yaw * yaw_err

            # ── Cost: control effort ────────────────────────────────────
            costs += self.w_effort * (v ** 2 + s ** 2)

            # ── Cost: control smoothness (penalize changes) ─────────────
            if t_step > 0:
                dv = v - speeds[:, t_step - 1]
                ds = s - steers[:, t_step - 1]
                costs += self.w_smooth * (dv ** 2 + ds ** 2)

            # ── Cost: progress — reward moving toward target ────────────
            if len(ref_arr) > 0:
                fwd_x_vec = np.cos(yaw)
                fwd_y_vec = np.sin(yaw)
                to_target_x = ref_arr[0, 0] - x
                to_target_y = ref_arr[0, 1] - y
                progress = fwd_x_vec * to_target_x + fwd_y_vec * to_target_y
                costs -= self.w_progress * np.clip(progress, 0.0, 2.0)

        # ── Terminal cost ───────────────────────────────────────────────
        if len(ref_arr) > 0:
            terminal_dist = np.sqrt(
                (x - ref_arr[-1, 0]) ** 2 +
                (y - ref_arr[-1, 1]) ** 2)
            costs += self.w_terminal * terminal_dist

        # ── Smoothness with previous command ────────────────────────────
        costs += self.w_smooth * 2.0 * (
            (speeds[:, 0] - self.prev_speed) ** 2 +
            (steers[:, 0] - self.prev_steering) ** 2)

        # ── Weighted average (softmax with temperature) ─────────────────
        costs -= np.min(costs)  # numerical stability
        weights = np.exp(-costs / max(self.lambda_temp, 0.01))
        weight_sum = np.sum(weights)

        if weight_sum < 1e-10:
            return 0.0, 0.0

        weights /= weight_sum

        best_speed = float(np.sum(weights * speeds[:, 0]))
        best_steer = float(np.sum(weights * steers[:, 0]))

        # Clamp final output
        best_speed = clamp(best_speed, -self.max_reverse_speed, self.max_speed)
        best_steer = clamp(best_steer, -self.max_steering_angle,
                           self.max_steering_angle)

        return best_speed, best_steer

    # ═════════════════════════════════════════════════════════════════════
    # Control Loop
    # ═════════════════════════════════════════════════════════════════════
    def _control_loop(self):
        """Main control loop at control_frequency Hz."""
        if not self.enabled:
            return

        # ── Get robot pose ──────────────────────────────────────────────
        pose = self._get_robot_pose()
        if pose is None:
            if self.stop_no_pose:
                self._publish_stop('no_pose')
            return
        rx, ry, ryaw = pose

        # ── Check semigoals available ───────────────────────────────────
        if not self.semigoals or self.current_sg_idx >= len(self.semigoals):
            if self.stop_no_sg:
                self._publish_stop('no_semigoals')
            return

        # ── Advance semigoal index ──────────────────────────────────────
        self._advance_semigoals(rx, ry, ryaw)

        if self.current_sg_idx >= len(self.semigoals):
            self._publish_stop('all_reached')
            self.get_logger().info('✅ All semigoals reached!',
                                   throttle_duration_sec=3.0)
            return

        # ── Get reference points for MPPI horizon ───────────────────────
        ref_points = self._get_reference_points()
        if not ref_points:
            self._publish_stop('no_refs')
            return

        # ── Run MPPI optimization ───────────────────────────────────────
        best_speed, best_steer = self._run_mppi(rx, ry, ryaw, ref_points)

        # ── Publish command ─────────────────────────────────────────────
        self._publish_cmd(best_steer, best_speed)
        self.prev_speed = best_speed
        self.prev_steering = best_steer

        # ── Debug ───────────────────────────────────────────────────────
        if self.publish_debug:
            sx, sy, _ = self.semigoals[self.current_sg_idx]
            dist = math.hypot(sx - rx, sy - ry)
            dbg = (f'sg={self.current_sg_idx}/{len(self.semigoals)} '
                   f'd={dist:.2f} v={best_speed:.2f} s={best_steer:.2f}')
            self.pub_debug.publish(String(data=dbg))
            self.pub_speed_dbg.publish(Float32(data=best_speed))
            self.pub_steer_dbg.publish(Float32(data=best_steer))
            self.pub_sg_idx_dbg.publish(
                Float32(data=float(self.current_sg_idx)))

    # ═════════════════════════════════════════════════════════════════════
    # Publishing helpers
    # ═════════════════════════════════════════════════════════════════════
    def _publish_cmd(self, steering, speed):
        """Publish MotorCommands to mixer input topic."""
        msg = MotorCommands()
        msg.motor_names = ['steering_angle', 'motor_throttle']
        msg.values = [float(steering), float(speed)]
        self.pub_cmd.publish(msg)

    def _publish_stop(self, reason=''):
        """Publish zero command (safe stop)."""
        self._publish_cmd(0.0, 0.0)
        self.prev_speed = 0.0
        self.prev_steering = 0.0
        if self.publish_debug and reason:
            self.get_logger().debug(
                f'STOP: {reason}', throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = MPPITrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Send stop before shutting down
        node._publish_stop('shutdown')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
