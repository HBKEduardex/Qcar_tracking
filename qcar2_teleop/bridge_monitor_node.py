#!/usr/bin/env python3
"""
Bridge Monitor — Dashboard display for the Pixel-Gated Hybrid Controller.

Uses ANSI cursor positioning for flicker-free dashboard-style updates.
Subscribes to /hybrid/* debug topics. Publishes /bridge_monitor/status.
"""
import sys
import json
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from geometry_msgs.msg import PoseStamped, Twist


# ── ANSI ────────────────────────────────────────────────────────────────
RST = '\033[0m'
BLD = '\033[1m'
DIM = '\033[2m'
RED = '\033[91m'
GRN = '\033[92m'
YEL = '\033[93m'
BLU = '\033[94m'
MAG = '\033[95m'
CYN = '\033[96m'
WHT = '\033[97m'
BGRED = '\033[41m'
BGGRN = '\033[42m'
BGYEL = '\033[43m'
BGMAG = '\033[45m'
HOME = '\033[H'
CLR  = '\033[J'
HIDE_CURSOR = '\033[?25l'
SHOW_CURSOR = '\033[?25h'


def bar(val, lo, hi, w=30, ch='█', em='░'):
    n = (val - lo) / max(hi - lo, 1e-9)
    n = max(0.0, min(1.0, n))
    f = int(n * w)
    return ch * f + em * (w - f)


class BridgeMonitorNode(Node):

    def __init__(self):
        super().__init__('bridge_monitor')
        self.declare_parameter('rate_hz', 10.0)
        rate = float(self.get_parameter('rate_hz').value)

        # State
        self.mode = -1.0
        self.fsm_state = '---'
        self.nav2_status = 'IDLE'
        self.yellow_px = 0.0
        self.blue_px = 0.0
        self.yellow_ratio = 0.0
        self.blue_ratio = 0.0
        self.gate_allowed = 0.0
        self.steering = 0.0
        self.speed = 0.0
        self.goal_index = '0/0'
        self.yaw_robot = 0.0
        self.yaw_goal = 0.0
        self.yaw_error = 0.0
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.cmd_lin = 0.0
        self.cmd_ang = 0.0
        self.prev_mode = -1.0
        self.history = []

        # Subs
        self.create_subscription(Float32, '/hybrid/mode', self._m, 10)
        self.create_subscription(String, '/hybrid/state', lambda m: setattr(self, 'fsm_state', m.data), 10)
        self.create_subscription(String, '/hybrid/nav2_status', lambda m: setattr(self, 'nav2_status', m.data), 10)
        self.create_subscription(Float32, '/hybrid/yellow_px', lambda m: setattr(self, 'yellow_px', m.data), 10)
        self.create_subscription(Float32, '/hybrid/blue_px', lambda m: setattr(self, 'blue_px', m.data), 10)
        self.create_subscription(Float32, '/hybrid/yellow_ratio', lambda m: setattr(self, 'yellow_ratio', m.data), 10)
        self.create_subscription(Float32, '/hybrid/blue_ratio', lambda m: setattr(self, 'blue_ratio', m.data), 10)
        self.create_subscription(Float32, '/hybrid/gate_allowed', lambda m: setattr(self, 'gate_allowed', m.data), 10)
        self.create_subscription(Float32, '/hybrid/steering', lambda m: setattr(self, 'steering', m.data), 10)
        self.create_subscription(Float32, '/hybrid/speed', lambda m: setattr(self, 'speed', m.data), 10)
        self.create_subscription(String, '/hybrid/goal_index', lambda m: setattr(self, 'goal_index', m.data), 10)
        self.create_subscription(Float32, '/hybrid/yaw_robot', lambda m: setattr(self, 'yaw_robot', m.data), 10)
        self.create_subscription(Float32, '/hybrid/yaw_goal', lambda m: setattr(self, 'yaw_goal', m.data), 10)
        self.create_subscription(Float32, '/hybrid/yaw_error', lambda m: setattr(self, 'yaw_error', m.data), 10)
        self.create_subscription(PoseStamped, '/hybrid/goal_pose', self._gp, 10)
        self.create_subscription(Twist, '/cmd_vel_nav', self._cv, 10)

        self.status_pub = self.create_publisher(String, '/bridge_monitor/status', 10)
        self.create_timer(1.0 / max(1.0, rate), self._draw)

        # Hide cursor + initial clear
        sys.stdout.write(HIDE_CURSOR + HOME + CLR)
        sys.stdout.flush()

    def _m(self, msg):
        n = msg.data
        if n != self.prev_mode and self.prev_mode is not None:
            o = 'PID' if self.prev_mode > 0.5 else ('NAV2' if self.prev_mode > -0.5 else 'STOP')
            nw = 'PID' if n > 0.5 else ('NAV2' if n > -0.5 else 'STOP')
            self.history.append(f'{time.strftime("%H:%M:%S")} {o}→{nw} G={self.goal_index}')
            if len(self.history) > 5:
                self.history.pop(0)
        self.prev_mode = n
        self.mode = n

    def _gp(self, m):
        self.goal_x = m.pose.position.x
        self.goal_y = m.pose.position.y

    def _cv(self, m):
        self.cmd_lin = m.linear.x
        self.cmd_ang = m.angular.z

    def destroy_node(self):
        sys.stdout.write(SHOW_CURSOR + '\n')
        sys.stdout.flush()
        super().destroy_node()

    def _draw(self):
        W = sys.stdout
        W.write(HOME)  # cursor to top-left, overwrite in place

        # ── Mode badge ──────────────────────────────────────────────────
        if self.mode > 0.5:
            mb = f'{BGGRN}{WHT}{BLD} ▶ PID  {RST}'
        elif self.mode > -0.5:
            mb = f'{BGYEL}{WHT}{BLD} ▶ NAV2 {RST}'
        else:
            mb = f'{BGRED}{WHT}{BLD} ■ STOP {RST}'

        # FSM badge
        fc = {
            'PID_ROAD': f'{BGGRN}{WHT}{BLD} PID_ROAD {RST}',
            'NAV2_INTERSECTION': f'{BGYEL}{WHT}{BLD} NAV2_INTERSECTION {RST}',
            'RECOVERY': f'{BGMAG}{WHT}{BLD} RECOVERY {RST}',
        }
        fb = fc.get(self.fsm_state, f'{DIM} {self.fsm_state} {RST}')

        # Gate
        gb = f'{YEL}🔓 OPEN{RST}' if self.gate_allowed > 0.5 else f'{GRN}🔒 CLOSED{RST}'

        lines = []
        a = lines.append

        a(f'{BLD}{CYN}╔══════════════════════════════════════════════════════════╗{RST}')
        a(f'{BLD}{CYN}║  HYBRID PIXEL-GATED MONITOR                            ║{RST}')
        a(f'{BLD}{CYN}╚══════════════════════════════════════════════════════════╝{RST}')
        a(f'')
        a(f'  FSM:  {fb}  Mode: {mb}  Gate: {gb}')
        # Nav2 status with BLIND_STRAIGHT highlight
        if self.nav2_status == 'BLIND_STRAIGHT':
            nav2_badge = f'\033[44m{WHT}{BLD} BLIND_STRAIGHT {RST}'
        else:
            nav2_badge = f'{CYN}{self.nav2_status:<12}{RST}'
        a(f'  Nav2: {nav2_badge}')
        a(f'')

        # Pixels
        a(f'{BLD}  ─── Pixel Gating ────────────────────────────────────────{RST}')
        yr = self.yellow_ratio
        br = self.blue_ratio
        yc = RED if yr < 0.04 else GRN
        bc = YEL if br > 0.6 else BLU
        yb = bar(yr, 0.0, 0.10, 30)
        bb = bar(br, 0.0, 1.0, 30)
        a(f'  {YEL}Yellow{RST} {yc}{int(self.yellow_px):>5}px{RST}  '
          f'{yc}{yr:.4f}{RST}  {DIM}│{RST}{yc}{yb}{RST}{DIM}│{RST} '
          f'{DIM}thr<0.04{RST}')
        a(f'  {BLU}Blue  {RST} {bc}{int(self.blue_px):>5}px{RST}  '
          f'{bc}{br:.4f}{RST}  {DIM}│{RST}{bc}{bb}{RST}{DIM}│{RST} '
          f'{DIM}thr>0.6{RST}')
        a(f'')

        # Mission
        a(f'{BLD}  ─── Mission ─────────────────────────────────────────────{RST}')
        a(f'  Goal: {CYN}{BLD}{self.goal_index}{RST}  '
          f'→ ({self.goal_x:.2f}, {self.goal_y:.2f})')
        a(f'')

        # Yaw
        a(f'{BLD}  ─── Yaw ─────────────────────────────────────────────────{RST}')
        ec = YEL if abs(self.yaw_error) > 1.0 else WHT
        eb = bar(abs(self.yaw_error), 0.0, 3.14, 20)
        a(f'  Robot: {self.yaw_robot:+.3f}rad  Goal: {self.yaw_goal:+.3f}rad')
        a(f'  Error: {ec}{self.yaw_error:+.3f}rad ({math.degrees(self.yaw_error):+.1f}°){RST} '
          f'{DIM}│{RST}{ec}{eb}{RST}{DIM}│{RST}')
        a(f'')

        # Motor
        a(f'{BLD}  ─── Motor ───────────────────────────────────────────────{RST}')
        sd = '◄' if self.steering < -0.01 else ('►' if self.steering > 0.01 else '●')
        sb = bar(abs(self.steering), 0.0, 0.45, 15)
        vb = bar(abs(self.speed), 0.0, 0.30, 15)
        a(f'  Steer: {sd} {self.steering:+.3f} {DIM}│{RST}{CYN}{sb}{RST}{DIM}│{RST}  '
          f'Speed: {self.speed:+.3f} {DIM}│{RST}{GRN}{vb}{RST}{DIM}│{RST}')
        a(f'  Nav2:  lin={CYN}{self.cmd_lin:+.3f}{RST}  ang={CYN}{self.cmd_ang:+.3f}{RST}')
        a(f'')

        # History
        if self.history:
            a(f'{BLD}  ─── Switch History ──────────────────────────────────────{RST}')
            for h in self.history:
                a(f'  {DIM}{h}{RST}')
            a(f'')

        a(f'{DIM}{"─" * 58}{RST}')

        # Pad to fill screen and prevent leftover lines
        output = '\n'.join(lines)
        # Clear from cursor to end of screen after writing
        W.write(output + CLR)
        W.flush()

        # Publish JSON
        s = json.dumps({
            'state': self.fsm_state,
            'mode': 'PID' if self.mode > 0.5 else ('NAV2' if self.mode > -0.5 else 'STOPPED'),
            'gate': self.gate_allowed > 0.5,
            'yellow_ratio': round(self.yellow_ratio, 4),
            'blue_ratio': round(self.blue_ratio, 4),
            'goal': self.goal_index,
        })
        self.status_pub.publish(String(data=s))


def main(args=None):
    rclpy.init(args=args)
    node = BridgeMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
