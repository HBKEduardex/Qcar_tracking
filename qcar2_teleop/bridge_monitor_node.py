#!/usr/bin/env python3
"""
Bridge Monitor — Visual debug for the Nav2 ↔ Lane Following hybrid.

Shows in real-time:
  - Current mode: LANE PID / NAV2 FALLBACK / STOPPED
  - When mode switches happen
  - Steering & speed from bridge
  - Nav2 cmd_vel inputs
  - Lane error & visibility
  - Curvature
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Twist
import time
import os


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
    DIM    = '\033[2m'


def bar(val, lo, hi, width=30, char='█', empty='░'):
    """Horizontal bar graph."""
    norm = (val - lo) / max(hi - lo, 1e-9)
    norm = max(0.0, min(1.0, norm))
    filled = int(norm * width)
    return char * filled + empty * (width - filled)


class BridgeMonitor(Node):
    def __init__(self):
        super().__init__('bridge_monitor')

        # State
        self.bridge_steer = 0.0
        self.bridge_speed = 0.0
        self.bridge_mode = -1.0  # -1 = no data yet
        self.nav2_linear = 0.0
        self.nav2_angular = 0.0
        self.lane_error = 0.0
        self.lane_visible = False
        self.curvature = 0.0
        self.prev_mode = -1.0
        self.mode_switch_log = []  # last N switches
        self.last_update = time.time()

        # Subscribers — hybrid_controller outputs (was /bridge/*, now /hybrid/*)
        self.create_subscription(Float32, '/hybrid/steering', self._cb_steer, 10)
        self.create_subscription(Float32, '/hybrid/speed', self._cb_speed, 10)
        self.create_subscription(Float32, '/hybrid/mode', self._cb_mode, 10)

        # Subscribers — Nav2 input
        self.create_subscription(Twist, '/cmd_vel_nav', self._cb_nav2, 10)

        # Subscribers — Lane data
        self.create_subscription(Float32, '/lane/center/error', self._cb_lane_err, 10)
        self.create_subscription(Bool, '/lane/center/visible', self._cb_lane_vis, 10)
        self.create_subscription(Float32, '/lane/curvature', self._cb_curv, 10)

        # Display timer (10 Hz refresh)
        self.create_timer(0.1, self._display)

        self.get_logger().info('Bridge Monitor started — watching hybrid_controller state (/hybrid/*)')

    # ─── Callbacks ───
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
            if len(self.mode_switch_log) > 8:
                self.mode_switch_log.pop(0)
        self.bridge_mode = new_mode
        self.prev_mode = new_mode

    def _cb_nav2(self, msg):
        self.nav2_linear = msg.linear.x
        self.nav2_angular = msg.angular.z

    def _cb_lane_err(self, msg):
        self.lane_error = msg.data

    def _cb_lane_vis(self, msg):
        self.lane_visible = msg.data

    def _cb_curv(self, msg):
        self.curvature = msg.data

    @staticmethod
    def _mode_name(m):
        if m >= 0.9:
            return 'LANE PID'
        elif m >= -0.5:
            return 'NAV2 TURN'
        return 'STOPPED'

    # ─── Display ───
    def _display(self):
        stale = (time.time() - self.last_update) > 1.0

        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')

        print(f"{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════╗{C.RESET}")
        print(f"{C.BOLD}{C.CYAN}║   🔀  HYBRID CONTROLLER MONITOR (/hybrid/*)     ║{C.RESET}")
        print(f"{C.BOLD}{C.CYAN}╚══════════════════════════════════════════════════╝{C.RESET}")
        print()

        # ─── Mode ───
        if stale or self.bridge_mode < -0.5:
            mode_str = f"{C.BG_RED}{C.BOLD}  ⏹  STOPPED / NO DATA  {C.RESET}"
        elif self.bridge_mode >= 0.9:
            mode_str = f"{C.BG_GRN}{C.BOLD}  🛣  LANE PID  {C.RESET}"
        else:
            mode_str = f"{C.BG_YEL}{C.BOLD}  🔄  NAV2 TURN (intersection)  {C.RESET}"

        print(f"  Mode:  {mode_str}")
        print()

        # ─── Bridge output ───
        print(f"{C.BOLD}  ── Bridge Output ──────────────────────────────{C.RESET}")
        steer_color = C.GREEN if abs(self.bridge_steer) < 0.1 else (C.YELLOW if abs(self.bridge_steer) < 0.3 else C.RED)
        steer_dir = "◀ LEFT " if self.bridge_steer > 0.01 else ("RIGHT ▶" if self.bridge_steer < -0.01 else "STRAIGHT")
        print(f"  Steering: {steer_color}{self.bridge_steer:+.4f}{C.RESET}  {steer_dir}")
        print(f"            {C.DIM}{bar(self.bridge_steer, -0.45, 0.45)}{C.RESET}")
        print(f"  Speed:    {C.CYAN}{self.bridge_speed:+.4f}{C.RESET} m/s")
        print(f"            {C.DIM}{bar(abs(self.bridge_speed), 0, 0.3)}{C.RESET}")
        print()

        # ─── Nav2 input ───
        print(f"{C.BOLD}  ── Nav2 Input (/cmd_vel_nav) ──────────────────{C.RESET}")
        nav_active = abs(self.nav2_linear) > 0.001
        nav_status = f"{C.GREEN}ACTIVE{C.RESET}" if nav_active else f"{C.RED}IDLE (no goal){C.RESET}"
        print(f"  Status:   {nav_status}")
        print(f"  Linear.x: {C.CYAN}{self.nav2_linear:+.4f}{C.RESET} m/s")
        ang_dir = "↶ LEFT" if self.nav2_angular > 0.05 else ("RIGHT ↷" if self.nav2_angular < -0.05 else "straight")
        print(f"  Angular.z:{C.MAG}{self.nav2_angular:+.4f}{C.RESET} rad/s  ({ang_dir})")
        print()

        # ─── Lane data ───
        print(f"{C.BOLD}  ── Lane Detection ────────────────────────────{C.RESET}")
        vis_str = f"{C.GREEN}✔ VISIBLE{C.RESET}" if self.lane_visible else f"{C.RED}✘ LOST{C.RESET}"
        print(f"  Visible:  {vis_str}")
        err_color = C.GREEN if abs(self.lane_error) < 0.15 else (C.YELLOW if abs(self.lane_error) < 0.4 else C.RED)
        err_dir = "← left" if self.lane_error > 0.05 else ("right →" if self.lane_error < -0.05 else "centered")
        print(f"  Error:    {err_color}{self.lane_error:+.4f}{C.RESET}  ({err_dir})")
        print(f"            {C.DIM}{bar(self.lane_error, -1.0, 1.0)}{C.RESET}")
        print(f"  Curvature:{C.BLUE}{self.curvature:+.6f}{C.RESET}")
        print()

        # ─── Mode switch log ───
        print(f"{C.BOLD}  ── Mode Switch History ────────────────────────{C.RESET}")
        if self.mode_switch_log:
            for entry in self.mode_switch_log[-6:]:
                print(f"  {C.DIM}{entry}{C.RESET}")
        else:
            print(f"  {C.DIM}(no switches yet){C.RESET}")
        print()

        # ─── Decision explanation ───
        print(f"{C.BOLD}  ── Why this mode? ────────────────────────────{C.RESET}")
        if stale or self.bridge_mode < -0.5:
            print(f"  {C.RED}No data from hybrid_controller. Is it running?{C.RESET}")
        elif not nav_active:
            print(f"  {C.DIM}Nav2 linear.x ≈ 0 → no goal active → stopped{C.RESET}")
        elif self.bridge_mode >= 0.9:
            print(f"  {C.GREEN}Lane visible + |angular.z| < threshold → LANE PID{C.RESET}")
            print(f"  {C.DIM}Speed from Nav2 ({self.nav2_linear:+.3f}) × curve/error factors{C.RESET}")
        else:
            print(f"  {C.YELLOW}INTERSECTION: |angular.z|={abs(self.nav2_angular):.3f} > threshold{C.RESET}")
            print(f"  {C.YELLOW}Nav2 controls steering → turn {ang_dir}{C.RESET}")
            print(f"  {C.DIM}Exits when |angular.z| drops below 0.15 rad/s{C.RESET}")


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
