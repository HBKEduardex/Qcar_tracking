#!/usr/bin/env python3
"""
Hybrid Switch Controller — Pixel-Gated Nav2 ↔ Lane Tracking PID

Switches between Nav2 and PID based on color-mask pixel counts:
  - Yellow pixels = lane markings (from segmentation)
  - Blue pixels   = road / intersection indicator

FSM States:
  PID_ROAD            → normal lane following
  NAV2_INTERSECTION   → Nav2 controls (intersection detected)
  RECOVERY            → cooldown after Nav2, transitioning back to PID

Gating rules:
  PID → NAV2:  yellow_ratio < yellow_low AND blue_ratio > blue_high
               sustained for activate_frames consecutive frames
  NAV2 → REC:  blue_ratio < blue_exit for reacquire_frames frames
  REC → PID:   cooldown_sec elapsed

Special case (rotonda):
  yellow_ratio low BUT blue_ratio also low → STAY PID

NAV2 MODE: sends goal to /goal_pose, bridges /cmd_vel_nav → motors (max steer).
PID MODE:  forwards /lane/motor_cmd → motors.
AUTO-RETRY: re-sends goal if Nav2 stops sending cmd_vel.

Subscribes:
    /segmentation/color_mask (Image BGR)  — color segmentation
    /mission_goals           (PoseStamped) — from planner
    /lane/motor_cmd          (MotorCommands) — PID output
    /cmd_vel_nav             (Twist)       — Nav2 velocity

Publishes:
    /goal_pose               (PoseStamped)   — for Nav2
    /qcar2_motor_speed_cmd   (MotorCommands) — motor commands
    /hybrid/mode             (Float32)       — 1=PID, 0=NAV2, -1=STOPPED
    /hybrid/state            (String)        — PID_ROAD / NAV2_INTERSECTION / RECOVERY
    /hybrid/yellow_px        (Float32)
    /hybrid/blue_px          (Float32)
    /hybrid/yellow_ratio     (Float32)
    /hybrid/blue_ratio       (Float32)
    /hybrid/gate_allowed     (Float32)       — 1.0 if Nav2 gate open
    /hybrid/steering         (Float32)
    /hybrid/speed            (Float32)
    /hybrid/goal_index       (String)
    /hybrid/yaw_error        (Float32)
    /hybrid/yaw_robot        (Float32)
    /hybrid/yaw_goal         (Float32)
    /hybrid/nav2_status      (String)
    /hybrid/goal_pose        (PoseStamped)
"""

import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32, String
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Twist
from qcar2_interfaces.msg import MotorCommands
from cv_bridge import CvBridge

import tf2_ros
from tf2_ros import Buffer, TransformListener


# ── FSM states ──────────────────────────────────────────────────────────
STATE_PID_ROAD          = 'PID_ROAD'
STATE_NAV2_INTERSECTION = 'NAV2_INTERSECTION'
STATE_RECOVERY          = 'RECOVERY'

# ── Mode codes (for /hybrid/mode topic) ─────────────────────────────────
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


def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


class HybridSwitchController(Node):

    def __init__(self):
        super().__init__('hybrid_switch_controller')

        # ── Parameters: pixel gating ────────────────────────────────────
        self.declare_parameter('mask_topic', '/lokita')
        self.declare_parameter('use_bottom_ratio', 0.45)
        self.declare_parameter('yellow_low_threshold', 0.04)   # more flexible
        self.declare_parameter('blue_high_threshold', 0.6)    # more flexible
        self.declare_parameter('blue_exit_threshold', 0.3)
        self.declare_parameter('activate_frames', 3)           # fast response
        self.declare_parameter('reacquire_frames', 5)
        self.declare_parameter('cooldown_sec', 3.0)
        # gate_mode: 'AND' = yellow_low AND blue_high
        #            'OR'  = yellow_low OR  blue_high  (most flexible)
        #            'YELLOW_ONLY' = just yellow_low
        self.declare_parameter('gate_mode', 'AND')

        # ── Parameters: navigation ──────────────────────────────────────
        self.declare_parameter('goal_tolerance', 0.3)
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('map_frame', 'pgm_map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('mission_goals_topic', '/mission_goals')
        self.declare_parameter('motor_cmd_topic', '/qcar2_motor_speed_cmd')
        self.declare_parameter('lane_cmd_topic', '/lane/motor_cmd')
        self.declare_parameter('goal_pose_topic', '/goal_pose')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel_nav')
        self.declare_parameter('pid_speed_override', 0.0)

        # ── Parameters: Nav2 bridging ───────────────────────────────────
        self.declare_parameter('nav2_speed_scale', 1.0)
        self.declare_parameter('max_angle', 0.45)
        self.declare_parameter('max_speed', 0.30)
        self.declare_parameter('nav2_timeout', 0.5)
        self.declare_parameter('retry_interval', 3.0)
        self.declare_parameter('yaw_error_threshold', 0.5)     # rad

        # ── Parameters: Blind-drive at intersections ────────────────────
        self.declare_parameter('blind_drive_speed', 0.20)      # m/s
        self.declare_parameter('blind_heading_kp', 0.5)        # yaw→steer gain
        self.declare_parameter('blind_max_steer', 0.15)        # rad (~8.5°)
        self.declare_parameter('blind_pid_blend', 0.3)         # max PID weight
        self.declare_parameter('blind_red_threshold', 0.065)    # red ratio below = no edges

        # ── Read all parameters ─────────────────────────────────────────
        mask_topic = str(self.get_parameter('mask_topic').value)
        self.use_bottom_ratio = float(self.get_parameter('use_bottom_ratio').value)
        self.yellow_low_thresh = float(self.get_parameter('yellow_low_threshold').value)
        self.blue_high_thresh = float(self.get_parameter('blue_high_threshold').value)
        self.blue_exit_thresh = float(self.get_parameter('blue_exit_threshold').value)
        self.activate_frames = int(self.get_parameter('activate_frames').value)
        self.reacquire_frames = int(self.get_parameter('reacquire_frames').value)
        self.cooldown_sec = float(self.get_parameter('cooldown_sec').value)
        self.gate_mode = str(self.get_parameter('gate_mode').value).upper()

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

        self.nav2_speed_scale = float(self.get_parameter('nav2_speed_scale').value)
        self.max_angle = float(self.get_parameter('max_angle').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.nav2_timeout = float(self.get_parameter('nav2_timeout').value)
        self.retry_interval = float(self.get_parameter('retry_interval').value)
        self.yaw_error_threshold = float(self.get_parameter('yaw_error_threshold').value)

        # Blind-drive
        self.blind_drive_speed = float(self.get_parameter('blind_drive_speed').value)
        self.blind_heading_kp = float(self.get_parameter('blind_heading_kp').value)
        self.blind_max_steer = float(self.get_parameter('blind_max_steer').value)
        self.blind_pid_blend = float(self.get_parameter('blind_pid_blend').value)
        self.blind_red_threshold = float(self.get_parameter('blind_red_threshold').value)

        # ── State: FSM ──────────────────────────────────────────────────
        self.fsm_state = STATE_PID_ROAD
        self.activate_counter = 0       # frames toward Nav2 activation
        self.reacquire_counter = 0      # frames toward Nav2 exit
        self.cooldown_start = None      # when RECOVERY started

        # ── State: pixel counts ─────────────────────────────────────────
        self.bridge_cv = CvBridge()
        self.yellow_px = 0
        self.blue_px = 0
        self.red_px = 0
        self.yellow_ratio = 0.0
        self.blue_ratio = 0.0
        self.red_ratio = 0.0
        self.gate_nav2_allowed = False
        self.mask_frame_count = 0

        # ── State: mission goals ────────────────────────────────────────
        self.mission_goals = []
        self.current_goal_idx = 0
        self.goal_sent_to_nav2 = False

        # ── State: Nav2 cmd_vel ─────────────────────────────────────────
        self.nav2_linear_x = 0.0
        self.nav2_angular_z = 0.0
        self.last_cmd_vel_time = self.get_clock().now()
        self.nav2_status = 'IDLE'
        self.last_retry_time = self.get_clock().now()
        self.retry_count = 0

        # ── State: Lane PID ─────────────────────────────────────────────
        self.lane_steering = 0.0
        self.lane_speed = 0.0

        # ── State: yaw tracking (debug) ─────────────────────────────────
        self.yaw_robot = 0.0
        self.yaw_goal = 0.0
        self.yaw_error = 0.0

        # ── State: output ───────────────────────────────────────────────
        self.active_steering = 0.0
        self.active_speed = 0.0
        self.current_mode = MODE_STOPPED

        # ── TF2 ─────────────────────────────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Subscribers ─────────────────────────────────────────────────
        self.create_subscription(Image, mask_topic, self._cb_mask, 10)
        self.create_subscription(PoseStamped, mission_topic, self._cb_mission_goal, 10)
        self.create_subscription(MotorCommands, lane_topic, self._cb_lane_cmd, 10)
        self.create_subscription(Twist, cmd_vel_topic, self._cb_cmd_vel, 10)

        # ── Publishers ──────────────────────────────────────────────────
        self.motor_pub = self.create_publisher(MotorCommands, motor_topic, 10)
        self.goal_pose_pub = self.create_publisher(PoseStamped, goal_pose_topic, 10)

        # Debug publishers
        self.pub_mode = self.create_publisher(Float32, '/hybrid/mode', 10)
        self.pub_state = self.create_publisher(String, '/hybrid/state', 10)
        self.pub_yellow_px = self.create_publisher(Float32, '/hybrid/yellow_px', 10)
        self.pub_blue_px = self.create_publisher(Float32, '/hybrid/blue_px', 10)
        self.pub_yellow_ratio = self.create_publisher(Float32, '/hybrid/yellow_ratio', 10)
        self.pub_blue_ratio = self.create_publisher(Float32, '/hybrid/blue_ratio', 10)
        self.pub_gate = self.create_publisher(Float32, '/hybrid/gate_allowed', 10)
        self.pub_steering = self.create_publisher(Float32, '/hybrid/steering', 10)
        self.pub_speed = self.create_publisher(Float32, '/hybrid/speed', 10)
        self.pub_goal_index = self.create_publisher(String, '/hybrid/goal_index', 10)
        self.pub_yaw_error = self.create_publisher(Float32, '/hybrid/yaw_error', 10)
        self.pub_yaw_robot = self.create_publisher(Float32, '/hybrid/yaw_robot', 10)
        self.pub_yaw_goal = self.create_publisher(Float32, '/hybrid/yaw_goal', 10)
        self.pub_nav2_status = self.create_publisher(String, '/hybrid/nav2_status', 10)
        self.pub_goal_pose = self.create_publisher(PoseStamped, '/hybrid/goal_pose', 10)

        # ── Timer ───────────────────────────────────────────────────────
        period = 1.0 / max(1.0, self.rate_hz)
        self.create_timer(period, self._control_loop)

        self.get_logger().info(
            f'HybridSwitchController [PIXEL-GATED] started\n'
            f'  gate_mode={self.gate_mode}, '
            f'yellow_low={self.yellow_low_thresh}, '
            f'blue_high={self.blue_high_thresh}, '
            f'blue_exit={self.blue_exit_thresh}\n'
            f'  activate_frames={self.activate_frames}, '
            f'reacquire_frames={self.reacquire_frames}, '
            f'cooldown={self.cooldown_sec}s\n'
            f'  goal_tol={self.goal_tolerance}m, '
            f'retry_interval={self.retry_interval}s'
        )

    # =====================================================================
    # Mask callback — compute yellow/blue pixel counts + update FSM gate
    # =====================================================================
    def _cb_mask(self, msg: Image):
        """Process segmentation mask: count yellow and blue pixels."""
        try:
            mask_bgr = self.bridge_cv.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'⚠️ CvBridge error: {e}', throttle_duration_sec=5.0)
            return

        if mask_bgr is None or mask_bgr.size == 0:
            return

        # Log first mask received
        if self.mask_frame_count == 0:
            self.get_logger().info(
                f'✅ First mask received! shape={mask_bgr.shape} '
                f'topic={self.get_parameter("mask_topic").value}'
            )

        H, W = mask_bgr.shape[:2]
        y0 = int(H * (1.0 - self.use_bottom_ratio))
        roi = mask_bgr[y0:, :]
        h, w = roi.shape[:2]
        roi_area = float(h * w) if (h * w) > 0 else 1.0

        # Yellow: HSV [20-40, 120-255, 120-255]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        yellow_bin = cv2.inRange(hsv,
                                 np.array([20, 120, 120], dtype=np.uint8),
                                 np.array([40, 255, 255], dtype=np.uint8))
        self.yellow_px = int(cv2.countNonZero(yellow_bin))
        self.yellow_ratio = self.yellow_px / roi_area

        # Blue (road): B>200, G<80, R<80
        b, g, r = cv2.split(roi)
        blue_bin = ((b > 200) & (g < 80) & (r < 80)).astype(np.uint8)
        self.blue_px = int(np.count_nonzero(blue_bin))
        self.blue_ratio = self.blue_px / roi_area

        # Red (edges): R>150, G<100, B<100  (same as yellow_line_position_node)
        red_bin = ((r > 150) & (g < 100) & (b < 100)).astype(np.uint8)
        self.red_px = int(np.count_nonzero(red_bin))
        self.red_ratio = self.red_px / roi_area

        # Periodic debug log
        self.mask_frame_count += 1
        if self.mask_frame_count % 40 == 0:
            self.get_logger().info(
                f'🎨 MASK #{self.mask_frame_count}: '
                f'yellow={self.yellow_px}px ({self.yellow_ratio:.4f}) '
                f'blue={self.blue_px}px ({self.blue_ratio:.4f}) | '
                f'gate_cond: y<{self.yellow_low_thresh}={self.yellow_ratio < self.yellow_low_thresh} '
                f'b>{self.blue_high_thresh}={self.blue_ratio > self.blue_high_thresh} | '
                f'act_cnt={self.activate_counter}/{self.activate_frames} | '
                f'state={self.fsm_state}'
            )

        # ── FSM gate update ─────────────────────────────────────────────
        yellow_low = self.yellow_ratio < self.yellow_low_thresh
        blue_high = self.blue_ratio > self.blue_high_thresh
        blue_exited = self.blue_ratio < self.blue_exit_thresh

        # Gate condition based on gate_mode
        if self.gate_mode == 'AND':
            gate_trigger = yellow_low and blue_high
        elif self.gate_mode == 'YELLOW_ONLY':
            gate_trigger = yellow_low
        else:  # 'OR' (default, most flexible)
            gate_trigger = yellow_low or blue_high

        if self.fsm_state == STATE_PID_ROAD:
            # Check activation
            if gate_trigger:
                self.activate_counter += 1
            else:
                self.activate_counter = 0

            if self.activate_counter >= self.activate_frames:
                self.fsm_state = STATE_NAV2_INTERSECTION
                self.activate_counter = 0
                self.reacquire_counter = 0
                self.goal_sent_to_nav2 = False  # force re-send
                self.get_logger().info(
                    f'🚦 FSM: PID_ROAD → NAV2_INTERSECTION | '
                    f'yellow={self.yellow_ratio:.4f} blue={self.blue_ratio:.4f} '
                    f'gate_mode={self.gate_mode} '
                    f'for {self.activate_frames} frames'
                )

            self.gate_nav2_allowed = False

        elif self.fsm_state == STATE_NAV2_INTERSECTION:
            # Exit Nav2 if: blue drops OR yellow comes back (lane re-detected)
            yellow_back = not yellow_low  # yellow_ratio >= threshold → lane visible
            should_exit = blue_exited or yellow_back

            if should_exit:
                self.reacquire_counter += 1
            else:
                self.reacquire_counter = 0

            if self.reacquire_counter >= self.reacquire_frames:
                reason = 'yellow_back' if yellow_back else 'blue_exit'
                self.fsm_state = STATE_RECOVERY
                self.cooldown_start = self.get_clock().now()
                self.reacquire_counter = 0
                self.get_logger().info(
                    f'🚦 FSM: NAV2_INTERSECTION → RECOVERY | '
                    f'reason={reason} '
                    f'yellow={self.yellow_ratio:.4f} blue={self.blue_ratio:.4f} '
                    f'for {self.reacquire_frames} frames'
                )

            self.gate_nav2_allowed = True

        elif self.fsm_state == STATE_RECOVERY:
            # Cooldown timer
            if self.cooldown_start is not None:
                dt = (self.get_clock().now() - self.cooldown_start).nanoseconds * 1e-9
                if dt >= self.cooldown_sec:
                    self.fsm_state = STATE_PID_ROAD
                    self.activate_counter = 0
                    self.cooldown_start = None
                    self.goal_sent_to_nav2 = False
                    self.get_logger().info(
                        f'🚦 FSM: RECOVERY → PID_ROAD | '
                        f'cooldown {self.cooldown_sec}s elapsed'
                    )

            self.gate_nav2_allowed = False

    # =====================================================================
    # Mission goal callback
    # =====================================================================
    def _cb_mission_goal(self, msg: PoseStamped):
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

    # =====================================================================
    # Lane PID callback
    # =====================================================================
    def _cb_lane_cmd(self, msg: MotorCommands):
        for i, name in enumerate(msg.motor_names):
            if name == 'steering_angle' and i < len(msg.values):
                self.lane_steering = float(msg.values[i])
            elif name == 'motor_throttle' and i < len(msg.values):
                self.lane_speed = float(msg.values[i])

    # =====================================================================
    # Nav2 cmd_vel callback
    # =====================================================================
    def _cb_cmd_vel(self, msg: Twist):
        self.nav2_linear_x = msg.linear.x
        self.nav2_angular_z = msg.angular.z
        self.last_cmd_vel_time = self.get_clock().now()

    # =====================================================================
    # Main control loop
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

        # ── Yaw error (debug + for display) ─────────────────────────────
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

            # Send next goal to Nav2 if in Nav2 mode
            if self.fsm_state == STATE_NAV2_INTERSECTION:
                next_goal = self.mission_goals[self.current_goal_idx]
                self._send_goal_pose(next_goal)

            self._publish_debug()
            return

        # ── MODE DECISION (based on FSM state) ──────────────────────────
        if self.fsm_state == STATE_NAV2_INTERSECTION:
            # ══════ INTERSECTION: yaw gate (quaternion-based) ══════════════

            # Yaw from quaternions: robot (TF) vs goal (PoseStamped)
            goal_yaw = quat_to_yaw(goal.pose.orientation)
            self.yaw_error = normalize_angle(goal_yaw - ryaw)
            self.yaw_goal = goal_yaw

            if abs(self.yaw_error) <= self.yaw_error_threshold:
                if self.goal_sent_to_nav2:
                    self.goal_sent_to_nav2 = False

                # Triple-signal: is PID blind at an intersection?
                yellow_low = self.yellow_ratio < self.yellow_low_thresh
                blue_high = self.blue_ratio > self.blue_high_thresh
                red_low = self.red_ratio < self.blind_red_threshold
                in_intersection_blind = yellow_low and blue_high and red_low

                if in_intersection_blind:
                    # ── BLIND_STRAIGHT: intersection, no lane data ────
                    steering, speed = self._blind_straight_output()
                else:
                    # ── PID_THRU: lane data available (original) ──────
                    self.current_mode = MODE_PID
                    self.nav2_status = 'PID_THRU'
                    steering = self.lane_steering
                    speed = self.lane_speed
                    if self.pid_speed_override > 0.0:
                        speed = self.pid_speed_override
                    self._publish_motor(steering, speed)
                    self.active_steering = steering
                    self.active_speed = speed

            else:
                # ── Yaw large → ACTIVATE Nav2 for the turn ────────────────

                # Record initial sign when Nav2 first activates
                if not self.goal_sent_to_nav2:
                    self._nav2_initial_sign = 1.0 if self.yaw_error >= 0 else -1.0
                    self.get_logger().info(
                        f'🔀 |yaw_err|={abs(self.yaw_error):.2f}rad '
                        f'({math.degrees(abs(self.yaw_error)):.0f}°) > '
                        f'{self.yaw_error_threshold} → Nav2 '
                        f'(sign={"+1" if self._nav2_initial_sign > 0 else "-1"})'
                    )
                    self._send_goal_pose(goal)

                # Check if correction is done: sign flipped or near zero
                current_sign = 1.0 if self.yaw_error >= 0 else -1.0
                sign_flipped = (hasattr(self, '_nav2_initial_sign') and
                                current_sign != self._nav2_initial_sign)
                near_zero = abs(self.yaw_error) < 0.1  # ~5.7°

                if sign_flipped or near_zero:
                    # Turn correction complete
                    reason = 'sign_flip' if sign_flipped else 'near_zero'
                    self.get_logger().info(
                        f'✅ Nav2 turn done ({reason}): '
                        f'yaw_err={self.yaw_error:.2f}rad'
                    )
                    self.goal_sent_to_nav2 = False

                    # Check if still blind after the turn
                    yellow_low = self.yellow_ratio < self.yellow_low_thresh
                    blue_high = self.blue_ratio > self.blue_high_thresh
                    red_low = self.red_ratio < self.blind_red_threshold
                    in_intersection_blind = yellow_low and blue_high and red_low

                    if in_intersection_blind:
                        # ── BLIND_STRAIGHT after turn ─────────────────
                        steering, speed = self._blind_straight_output()
                    else:
                        # ── PID_THRU: lane data available (original) ──
                        self.current_mode = MODE_PID
                        self.nav2_status = 'PID_THRU'
                        steering = self.lane_steering
                        speed = self.lane_speed
                        if self.pid_speed_override > 0.0:
                            speed = self.pid_speed_override
                        self._publish_motor(steering, speed)
                        self.active_steering = steering
                        self.active_speed = speed

                else:
                    # Still correcting → bridge Nav2
                    self.current_mode = MODE_NAV2

                    now = self.get_clock().now()
                    dt_cmd = (now - self.last_cmd_vel_time).nanoseconds * 1e-9

                    if dt_cmd > self.nav2_timeout:
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
                                f'{self.current_goal_idx + 1}/{len(self.mission_goals)}'
                            )
                        else:
                            self.nav2_status = 'NO_CMD_VEL'
                    else:
                        self.nav2_status = 'ACTIVE'
                        if abs(self.nav2_angular_z) > 0.01:
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
            # ═══════════════ PID MODE (PID_ROAD or RECOVERY) ═════════════
            self.current_mode = MODE_PID
            self.nav2_status = 'IDLE'

            # Reset Nav2 state
            if self.goal_sent_to_nav2:
                self.goal_sent_to_nav2 = False

            steering = self.lane_steering
            speed = self.lane_speed
            if self.pid_speed_override > 0.0:
                speed = self.pid_speed_override
            self._publish_motor(steering, speed)
            self.active_steering = steering
            self.active_speed = speed

        self._publish_debug()

    # =====================================================================
    # BLIND_STRAIGHT helper — drive straight with heading correction
    # =====================================================================
    def _blind_straight_output(self):
        """Compute steering+speed for blind intersection crossing.

        Uses existing self.yellow_ratio, self.yaw_error, self.lane_steering.
        Returns (steering, speed) and publishes motor command.
        """
        self.current_mode = MODE_PID
        self.nav2_status = 'BLIND_STRAIGHT'

        # Heading correction toward goal (gentle proportional)
        heading_steer = clamp(
            -self.blind_heading_kp * self.yaw_error,
            -self.blind_max_steer, self.blind_max_steer)

        # Gradual blend: as yellow_ratio rises toward threshold,
        # mix in more PID steering for a smooth transition
        blend_raw = self.yellow_ratio / max(self.yellow_low_thresh, 1e-6)
        blend = clamp(blend_raw * self.blind_pid_blend, 0.0, self.blind_pid_blend)
        steering = (1.0 - blend) * heading_steer + blend * self.lane_steering
        speed = self.blind_drive_speed

        self._publish_motor(steering, speed)
        self.active_steering = steering
        self.active_speed = speed

        return steering, speed

    # =====================================================================
    # Send goal to Nav2 via /goal_pose
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
    # Motor publishing
    # =====================================================================
    def _publish_motor(self, steering: float, speed: float):
        msg = MotorCommands()
        msg.motor_names = ['steering_angle', 'motor_throttle']
        msg.values = [float(steering), float(speed)]
        self.motor_pub.publish(msg)

    # =====================================================================
    # Debug publishing
    # =====================================================================
    def _publish_debug(self):
        self.pub_mode.publish(Float32(data=float(self.current_mode)))
        self.pub_state.publish(String(data=self.fsm_state))
        self.pub_yellow_px.publish(Float32(data=float(self.yellow_px)))
        self.pub_blue_px.publish(Float32(data=float(self.blue_px)))
        self.pub_yellow_ratio.publish(Float32(data=float(self.yellow_ratio)))
        self.pub_blue_ratio.publish(Float32(data=float(self.blue_ratio)))
        self.pub_gate.publish(Float32(data=1.0 if self.gate_nav2_allowed else 0.0))
        self.pub_yaw_error.publish(Float32(data=float(self.yaw_error)))
        self.pub_yaw_robot.publish(Float32(data=float(self.yaw_robot)))
        self.pub_yaw_goal.publish(Float32(data=float(self.yaw_goal)))
        self.pub_nav2_status.publish(String(data=self.nav2_status))
        self.pub_steering.publish(Float32(data=float(self.active_steering)))
        self.pub_speed.publish(Float32(data=float(self.active_speed)))

        total = len(self.mission_goals)
        idx = min(self.current_goal_idx + 1, total)
        self.pub_goal_index.publish(String(data=f'{idx}/{total}'))

        if self.mission_goals and self.current_goal_idx < len(self.mission_goals):
            g = self.mission_goals[self.current_goal_idx]
            p = PoseStamped()
            p.header.stamp = self.get_clock().now().to_msg()
            p.header.frame_id = self.map_frame
            p.pose = g.pose
            self.pub_goal_pose.publish(p)

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
