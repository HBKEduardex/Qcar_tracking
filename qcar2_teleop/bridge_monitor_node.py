#!/usr/bin/env python3
"""
Bridge Monitor — Visual debug for the Hybrid Switch Controller.

Shows in real-time:
  - Current mode: PID / NAV2 / STOPPED
  - Yaw robot, yaw goal, yaw error, threshold
  - Mission goal progress (1/6, 2/6, etc.)
  - Nav2 status (IDLE, ACTIVE, SUCCEEDED, ABORTED)
  - Steering & speed from controller
  - Nav2 cmd_vel inputs
  - Lane error & visibility
  - Mode switch history

Subscribes to /hybrid/* topics from hybrid_switch_controller_node.

Optionally publishes /bridge_monitor/status (JSON string).
"""

import json
import math
import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool, String
from geometry_msgs.msg import Twist, PoseStamped


# ANSI colors
class C:
    RESET  = '\033[0m'
    BOLD   = '\033[1m'
    RED    = '\033[91m'
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    BLUE   = '\033[94m'
    CYAN   = '\033[96m'
    MAG    = '\033[95m'
    BG_RED = '\033[41m'
    BG_GRN = '\033[42m'
    BG_YEL = '\033[43m'
    BG_BLU = '\033[44m'
    BG_CYN = '\033[46m'
    DIM    = '\033[2m'
    WHITE  = '\033[97m'


def bar(val, lo, hi, width=30, char='█', empty='░'):
    """Horizontal bar graph."""
    norm = (val - lo) / max(hi - lo, 1e-9)
    norm = max(0.0, min(1.0, norm))
    filled = int(norm * width)
    return char * filled + empty * (width - filled)


class BridgeMonitor(Node):
    def __init__(self):
        super().__init__('bridge_monitor')

        # ── State ───────────────────────────────────────────────────────────
        # Hybrid controller data
        self.bridge_steer = 0.0
        self.bridge_speed = 0.0
        self.bridge_mode = -1.0         # -1=STOPPED, 0=NAV2, 1=PID
        self.yaw_robot = 0.0
        self.yaw_goal = 0.0
        self.yaw_error = 0.0
        self.yaw_threshold = 0.5
        self.goal_index = '0/0'
        self.nav2_status = 'IDLE'
        self.goal_pose_x = 0.0
        self.goal_pose_y = 0.0
        self.goal_pose_yaw = 0.0

        # Nav2 cmd_vel
        self.nav2_linear = 0.0
        self.nav2_angular = 0.0

        # Lane data
        self.lane_error = 0.0
        self.lane_visible = False
        self.curvature = 0.0

        # History
        self.prev_mode = -1.0
        self.mode_switch_log = []
        self.last_update = time.time()
        self.abort_log = []             # recent Nav2 abort entries

        # ── Subscribers: Hybrid Switch Controller (/hybrid/*) ───────────────
        self.create_subscription(Float32, '/hybrid/steering', self._cb_steer, 10)
        self.create_subscription(Float32, '/hybrid/speed', self._cb_speed, 10)
        self.create_subscription(Float32, '/hybrid/mode', self._cb_mode, 10)
        self.create_subscription(Float32, '/hybrid/yaw_error', self._cb_yaw_error, 10)
        self.create_subscription(Float32, '/hybrid/yaw_robot', self._cb_yaw_robot, 10)
        self.create_subscription(Float32, '/hybrid/yaw_goal', self._cb_yaw_goal, 10)
        self.create_subscription(Float32, '/hybrid/yaw_threshold', self._cb_yaw_thresh, 10)
        self.create_subscription(String, '/hybrid/goal_index', self._cb_goal_idx, 10)
        self.create_subscription(String, '/hybrid/nav2_status', self._cb_nav2_status, 10)
        self.create_subscription(PoseStamped, '/hybrid/goal_pose', self._cb_goal_pose, 10)

        # ── Subscribers: Nav2 input ─────────────────────────────────────────
        self.create_subscription(Twist, '/cmd_vel_nav', self._cb_nav2, 10)

        # ── Subscribers: Lane data ──────────────────────────────────────────
        self.create_subscription(Float32, '/lane/center/error', self._cb_lane_err, 10)
        self.create_subscription(Bool, '/lane/center/visible', self._cb_lane_vis, 10)
        self.create_subscription(Float32, '/lane/curvature', self._cb_curv, 10)

        # ── Publisher: structured status ────────────────────────────────────
        self.status_pub = self.create_publisher(String, '/bridge_monitor/status', 10)

        # ── Display timer (10 Hz refresh) ───────────────────────────────────
        self.create_timer(0.1, self._display)

        self.get_logger().info(
            'Bridge Monitor started — watching hybrid_switch_controller (/hybrid/*)'
        )

    # ─── Callbacks: Hybrid controller ───────────────────────────────────────
    def _cb_steer(self, msg):
        self.bridge_steer = msg.data
        self.last_update = time.time()

    def _cb_speed(self, msg):
        self.bridge_speed = msg.data

    def _cb_mode(self, msg):
        new_mode = msg.data
        if self.prev_mode >= -0.5 and abs(new_mode - self.prev_mode) > 0.3:
            ts = time.strftime('%H:%M:%S')
            old_name = self._mode_name(self.prev_mode)
            new_name = self._mode_name(new_mode)
            entry = f"[{ts}] {old_name} → {new_name}"
            self.mode_switch_log.append(entry)
            if len(self.mode_switch_log) > 10:
                self.mode_switch_log.pop(0)
        self.bridge_mode = new_mode
        self.prev_mode = new_mode

    def _cb_yaw_error(self, msg):
        self.yaw_error = msg.data

    def _cb_yaw_robot(self, msg):
        self.yaw_robot = msg.data

    def _cb_yaw_goal(self, msg):
        self.yaw_goal = msg.data

    def _cb_yaw_thresh(self, msg):
        self.yaw_threshold = msg.data

    def _cb_goal_idx(self, msg):
        self.goal_index = msg.data

    def _cb_nav2_status(self, msg):
        prev = self.nav2_status
        self.nav2_status = msg.data
        # Log aborts
        if msg.data == 'ABORTED' and prev != 'ABORTED':
            ts = time.strftime('%H:%M:%S')
            entry = (
                f"[{ts}] ABORTED Goal {self.goal_index} | "
                f"Pose=({self.goal_pose_x:.2f}, {self.goal_pose_y:.2f}, "
                f"yaw={self.goal_pose_yaw:.2f})"
            )
            self.abort_log.append(entry)
            if len(self.abort_log) > 5:
                self.abort_log.pop(0)

    def _cb_goal_pose(self, msg):
        self.goal_pose_x = msg.pose.position.x
        self.goal_pose_y = msg.pose.position.y
        q = msg.pose.orientation
        self.goal_pose_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    # ─── Callbacks: Nav2 ────────────────────────────────────────────────────
    def _cb_nav2(self, msg):
        self.nav2_linear = msg.linear.x
        self.nav2_angular = msg.angular.z

    # ─── Callbacks: Lane ────────────────────────────────────────────────────
    def _cb_lane_err(self, msg):
        self.lane_error = msg.data

    def _cb_lane_vis(self, msg):
        self.lane_visible = msg.data

    def _cb_curv(self, msg):
        self.curvature = msg.data

    # ─── Helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _mode_name(m):
        if m >= 0.9:
            return 'PID'
        elif m >= -0.5:
            return 'NAV2'
        return 'STOPPED'

    # ─── Display ────────────────────────────────────────────────────────────
    def _display(self):
        stale = (time.time() - self.last_update) > 2.0

        # Publish structured status
        self._publish_status()

        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')

        print(f"{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════════════╗{C.RESET}")
        print(f"{C.BOLD}{C.CYAN}║   🔀  HYBRID SWITCH CONTROLLER MONITOR                 ║{C.RESET}")
        print(f"{C.BOLD}{C.CYAN}╚══════════════════════════════════════════════════════════╝{C.RESET}")
        print()

        # ─── Mode ───────────────────────────────────────────────────────
        if stale or self.bridge_mode < -0.5:
            mode_str = f"{C.BG_RED}{C.BOLD}{C.WHITE}  ⏹  STOPPED / NO DATA  {C.RESET}"
        elif self.bridge_mode >= 0.9:
            mode_str = f"{C.BG_GRN}{C.BOLD}  🛣  PID (Lane Following)  {C.RESET}"
        else:
            mode_str = f"{C.BG_BLU}{C.BOLD}{C.WHITE}  🧭  NAV2 (NavigateToPose)  {C.RESET}"

        print(f"  Mode:  {mode_str}")
        print()

        # ─── Mission Goal Progress ──────────────────────────────────────
        print(f"{C.BOLD}  ── Mission Goal ──────────────────────────────────────{C.RESET}")
        print(f"  Progress: {C.BOLD}{C.CYAN}Goal {self.goal_index}{C.RESET}")
        print(f"  Pose:     ({C.YELLOW}{self.goal_pose_x:+.3f}{C.RESET}, "
              f"{C.YELLOW}{self.goal_pose_y:+.3f}{C.RESET}, "
              f"yaw={C.YELLOW}{self.goal_pose_yaw:+.3f}{C.RESET} rad)")
        print()

        # ─── Yaw Analysis (CRITICAL SECTION) ────────────────────────────
        print(f"{C.BOLD}  ── Angular Analysis (Switch Decision) ────────────────{C.RESET}")
        print(f"  yaw_robot    = {C.CYAN}{self.yaw_robot:+.4f}{C.RESET} rad "
              f"({C.DIM}{math.degrees(self.yaw_robot):+.1f}°{C.RESET})")
        print(f"  yaw_goal     = {C.CYAN}{self.yaw_goal:+.4f}{C.RESET} rad "
              f"({C.DIM}{math.degrees(self.yaw_goal):+.1f}°{C.RESET})")

        yaw_err_raw = self.yaw_goal - self.yaw_robot
        print(f"  yaw_err_raw  = {C.MAG}{yaw_err_raw:+.4f}{C.RESET} rad")

        # Normalized error
        err_color = C.GREEN if abs(self.yaw_error) <= self.yaw_threshold else C.RED
        print(f"  yaw_error    = {err_color}{self.yaw_error:+.4f}{C.RESET} rad "
              f"({C.DIM}{math.degrees(self.yaw_error):+.1f}°{C.RESET})")
        print(f"  |yaw_error|  = {err_color}{abs(self.yaw_error):.4f}{C.RESET}")
        print(f"  threshold    = {C.YELLOW}{self.yaw_threshold:.4f}{C.RESET} rad "
              f"({C.DIM}{math.degrees(self.yaw_threshold):.1f}°{C.RESET})")

        # Visual bar: yaw_error vs threshold
        err_bar_val = abs(self.yaw_error)
        err_bar_max = max(self.yaw_threshold * 2.0, 1.0)
        thresh_pos = int(self.yaw_threshold / err_bar_max * 30)
        err_fill = int(min(err_bar_val / err_bar_max, 1.0) * 30)
        bar_chars = ''
        for i in range(30):
            if i < err_fill:
                bar_chars += f"{C.GREEN}█{C.RESET}" if i < thresh_pos else f"{C.RED}█{C.RESET}"
            elif i == thresh_pos:
                bar_chars += f"{C.YELLOW}│{C.RESET}"
            else:
                bar_chars += f"{C.DIM}░{C.RESET}"
        decision = 'PID' if abs(self.yaw_error) <= self.yaw_threshold else 'NAV2'
        print(f"              {bar_chars}  {decision}")
        print()

        # ─── Nav2 Status ────────────────────────────────────────────────
        print(f"{C.BOLD}  ── Nav2 Status ───────────────────────────────────────{C.RESET}")
        status_colors = {
            'IDLE': C.DIM, 'ACTIVE': C.GREEN, 'SUCCEEDED': C.CYAN,
            'ABORTED': C.RED, 'CANCELED': C.YELLOW, 'SENDING': C.MAG,
            'REJECTED': C.RED, 'ERROR': C.RED,
        }
        sc = status_colors.get(self.nav2_status, C.DIM)
        print(f"  Status:   {sc}{C.BOLD}{self.nav2_status}{C.RESET}")

        nav_active = abs(self.nav2_linear) > 0.001
        nav_label = f"{C.GREEN}ACTIVE{C.RESET}" if nav_active else f"{C.DIM}IDLE{C.RESET}"
        print(f"  cmd_vel:  Linear={C.CYAN}{self.nav2_linear:+.3f}{C.RESET} m/s  "
              f"Angular={C.MAG}{self.nav2_angular:+.3f}{C.RESET} rad/s  [{nav_label}]")
        print()

        # ─── Bridge Output ──────────────────────────────────────────────
        print(f"{C.BOLD}  ── Motor Output ──────────────────────────────────────{C.RESET}")
        steer_color = C.GREEN if abs(self.bridge_steer) < 0.1 else (
            C.YELLOW if abs(self.bridge_steer) < 0.3 else C.RED)
        steer_dir = "◀ LEFT " if self.bridge_steer > 0.01 else (
            "RIGHT ▶" if self.bridge_steer < -0.01 else "STRAIGHT")
        print(f"  Steering: {steer_color}{self.bridge_steer:+.4f}{C.RESET}  {steer_dir}")
        print(f"            {C.DIM}{bar(self.bridge_steer, -0.45, 0.45)}{C.RESET}")
        print(f"  Speed:    {C.CYAN}{self.bridge_speed:+.4f}{C.RESET} m/s")
        print(f"            {C.DIM}{bar(abs(self.bridge_speed), 0, 0.3)}{C.RESET}")
        print()

        # ─── Lane Data ──────────────────────────────────────────────────
        print(f"{C.BOLD}  ── Lane Detection ────────────────────────────────────{C.RESET}")
        vis_str = f"{C.GREEN}✔ VISIBLE{C.RESET}" if self.lane_visible else f"{C.RED}✘ LOST{C.RESET}"
        print(f"  Visible:  {vis_str}")
        err_color2 = C.GREEN if abs(self.lane_error) < 0.15 else (
            C.YELLOW if abs(self.lane_error) < 0.4 else C.RED)
        print(f"  Error:    {err_color2}{self.lane_error:+.4f}{C.RESET}")
        print(f"  Curvature:{C.BLUE}{self.curvature:+.6f}{C.RESET}")
        print()

        # ─── Mode Switch Log ───────────────────────────────────────────
        print(f"{C.BOLD}  ── Mode Switch History ────────────────────────────────{C.RESET}")
        if self.mode_switch_log:
            for entry in self.mode_switch_log[-6:]:
                print(f"  {C.DIM}{entry}{C.RESET}")
        else:
            print(f"  {C.DIM}(no switches yet){C.RESET}")
        print()

        # ─── Abort Log ──────────────────────────────────────────────────
        if self.abort_log:
            print(f"{C.BOLD}{C.RED}  ── Nav2 ABORT Log ────────────────────────────────────{C.RESET}")
            for entry in self.abort_log[-4:]:
                print(f"  {C.RED}{entry}{C.RESET}")
            print()

        # ─── Decision Explanation ───────────────────────────────────────
        print(f"{C.BOLD}  ── Why this mode? ────────────────────────────────────{C.RESET}")
        if stale or self.bridge_mode < -0.5:
            print(f"  {C.RED}No data from hybrid_switch_controller. Is it running?{C.RESET}")
        elif self.bridge_mode >= 0.9:
            print(f"  {C.GREEN}|yaw_error|={abs(self.yaw_error):.3f} ≤ "
                  f"threshold={self.yaw_threshold:.3f} → PID lane following{C.RESET}")
        else:
            print(f"  {C.BLUE}|yaw_error|={abs(self.yaw_error):.3f} > "
                  f"threshold={self.yaw_threshold:.3f} → Nav2 turning{C.RESET}")

    # ─── Structured status publisher ────────────────────────────────────────
    def _publish_status(self):
        """Publish a JSON status message on /bridge_monitor/status."""
        status = {
            'mode': self._mode_name(self.bridge_mode),
            'goal_index': self.goal_index,
            'yaw_robot': round(self.yaw_robot, 4),
            'yaw_goal': round(self.yaw_goal, 4),
            'yaw_error': round(self.yaw_error, 4),
            'abs_yaw_error': round(abs(self.yaw_error), 4),
            'yaw_threshold': round(self.yaw_threshold, 4),
            'nav2_status': self.nav2_status,
            'steering': round(self.bridge_steer, 4),
            'speed': round(self.bridge_speed, 4),
            'goal_pose': {
                'x': round(self.goal_pose_x, 3),
                'y': round(self.goal_pose_y, 3),
                'yaw': round(self.goal_pose_yaw, 3),
            },
            'lane_visible': self.lane_visible,
            'lane_error': round(self.lane_error, 4),
        }
        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BridgeMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
