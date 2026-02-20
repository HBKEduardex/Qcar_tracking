#!/usr/bin/env python3
"""
Hybrid Nav2 ↔ Lane Following Controller (v5 — with Forced Nav2 Mode)

Architecture:
  - Receives MotorCommands from yellow_line_follower_controller via /lane/motor_cmd
  - Receives cmd_vel from Nav2 via /cmd_vel_nav
  - Receives exploration goals from exploration_manager_node via /exploration_goal
  - Monitors lane detection quality via /lane/yellow/visible and /lane/edge/count
  - Forces Nav2 control when lane detection is insufficient

Behavior:
  - Before receiving first exploration goal: LANE_ONLY mode
  - After receiving first exploration goal: HYBRID mode with priority system

Priority System for Forced Nav2:
  - Force Nav2 when: only 1 edge (no yellow, no 2 edges) OR nothing detected (all black)
  - Return to lane follower when: (yellow visible AND 1+ edge) OR (2+ edges)

Modes:
  1. LANE_ONLY   (mode=2.0) — steering+speed from lane follower only (pre-goal)
  2. LANE_PID    (mode=1.0) — steering from lane follower + speed from Nav2
  3. NAV2_TURN   (mode=0.0) — steering from Nav2 angular.z (intersection)
  4. NAV2_FORCED (mode=3.0) — Nav2 controls all (poor lane detection)
  5. STOPPED     (mode=-1)  — no commands or Nav2 says stop

Subscribes:
  /cmd_vel_nav         (geometry_msgs/Twist)            — Nav2 velocity command
  /lane/motor_cmd      (qcar2_interfaces/MotorCommands) — steering+speed from lane follower
  /lane/center/visible (std_msgs/Bool)                  — lane center detected?
  /lane/curvature      (std_msgs/Float32)               — lane curvature
  /exploration_goal    (geometry_msgs/PoseStamped)      — goals from exploration manager
  /lane/yellow/visible (std_msgs/Bool)                  — yellow line detected?
  /lane/edge/count     (std_msgs/Float32)               — number of edges detected (0,1,2)

Publishes:
  /qcar2_motor_speed_cmd (qcar2_interfaces/MotorCommands) — final motor commands
  /goal_pose             (geometry_msgs/PoseStamped)      — goals forwarded to Nav2
  /hybrid/steering       (std_msgs/Float32)               — debug: current steering (rad)
  /hybrid/speed          (std_msgs/Float32)               — debug: current speed (m/s)
  /hybrid/mode           (std_msgs/Float32)               — debug: mode indicator
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Twist, PoseStamped
from qcar2_interfaces.msg import MotorCommands
from rcl_interfaces.msg import SetParametersResult
import math


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# Modes
MODE_STOPPED    = -1.0
MODE_NAV2_TURN  =  0.0
MODE_LANE_PID   =  1.0
MODE_LANE_ONLY  =  2.0  # Solo lane follower (antes de recibir primer goal)
MODE_NAV2_FORCED = 3.0  # Forzar Nav2 cuando detección de carril es insuficiente


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

        # ── Exploration goal topics ──
        self.declare_parameter('exploration_goal_topic', '/exploration_goal')
        self.declare_parameter('nav2_goal_topic', '/goal_pose')

        # ── Lane detection topics for forced Nav2 ──
        self.declare_parameter('yellow_visible_topic', '/lane/yellow/visible')
        self.declare_parameter('edge_count_topic', '/lane/edge/count')
        self.declare_parameter('edge_position_topic', '/lane/edge/position')  # Normalized edge position
        
        # ── Forced Nav2 parameters ──
        self.declare_parameter('nav2_forced_speed_scale', 0.7)  # Reduce speed when forcing Nav2
        self.declare_parameter('nav2_forced_steer_gain', 0.8)   # Steering gain when forcing Nav2
        self.declare_parameter('min_edges_for_lane', 2)         # Min edges needed to trust lane follower
        self.declare_parameter('require_yellow_for_lane', True) # Require yellow line for lane mode
        
        # ── Edge safety parameters (avoid crossing red line) ──
        self.declare_parameter('edge_safety_enabled', True)     # Enable edge crossing prevention
        self.declare_parameter('edge_safety_threshold', 0.6)    # Edge position threshold (0-1, 1=edge of image)
        self.declare_parameter('edge_safety_steer_limit', 0.3)  # Max steering towards edge (rad)
        
        # ── Hybrid anti-local-minima parameters ──
        self.declare_parameter('hybrid_lane_speed_factor', 0.8) # Factor for lane speed in hybrid mode
        self.declare_parameter('hybrid_use_max_speed', True)    # True=use max(nav2,lane), False=use nav2 only
        
        # ── Hybrid steering blend parameters ──
        self.declare_parameter('hybrid_nav2_steer_weight', 0.7)  # Weight for Nav2 steering in hybrid mode (0-1)
        self.declare_parameter('hybrid_lane_steer_weight', 0.3)  # Weight for lane steering in hybrid mode (0-1)

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

        self.exploration_goal_topic = self.get_parameter('exploration_goal_topic').value
        self.nav2_goal_topic = self.get_parameter('nav2_goal_topic').value

        self.yellow_visible_topic = self.get_parameter('yellow_visible_topic').value
        self.edge_count_topic = self.get_parameter('edge_count_topic').value
        self.edge_position_topic = self.get_parameter('edge_position_topic').value
        self.nav2_forced_speed_scale = float(self.get_parameter('nav2_forced_speed_scale').value)
        self.nav2_forced_steer_gain = float(self.get_parameter('nav2_forced_steer_gain').value)
        self.min_edges_for_lane = int(self.get_parameter('min_edges_for_lane').value)
        self.require_yellow_for_lane = bool(self.get_parameter('require_yellow_for_lane').value)
        self.hybrid_lane_speed_factor = float(self.get_parameter('hybrid_lane_speed_factor').value)
        self.hybrid_use_max_speed = bool(self.get_parameter('hybrid_use_max_speed').value)
        self.hybrid_nav2_steer_weight = float(self.get_parameter('hybrid_nav2_steer_weight').value)
        self.hybrid_lane_steer_weight = float(self.get_parameter('hybrid_lane_steer_weight').value)
        
        # Edge safety parameters
        self.edge_safety_enabled = bool(self.get_parameter('edge_safety_enabled').value)
        self.edge_safety_threshold = float(self.get_parameter('edge_safety_threshold').value)
        self.edge_safety_steer_limit = float(self.get_parameter('edge_safety_steer_limit').value)

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

        # Exploration goal state — si no ha recibido goal, usa solo lane follower
        self.has_received_goal = False
        self.current_goal = None

        # Lane detection state for forced Nav2
        self.yellow_visible = False       # Línea amarilla detectada
        self.edge_count = 0               # Número de bordes rojos detectados (0, 1, 2)
        self.force_nav2 = False           # Flag: forzar Nav2 por detección insuficiente
        self.detection_quality = 'NONE'   # NONE, PARTIAL_EDGE, PARTIAL_YELLOW, ACCEPTABLE, GOOD
        self.edge_position = 999.0        # Posición normalizada del borde rojo (-1 a 1, 999=sin dato)
        self.last_lane_only_warn = self.get_clock().now()  # Throttle para warnings

        # ─── Subs ───
        self.sub_cmd_vel = self.create_subscription(
            Twist, '/cmd_vel_nav', self.cb_cmd_vel, 10)
        self.sub_lane_cmd = self.create_subscription(
            MotorCommands, self.lane_cmd_topic, self.cb_lane_cmd, 10)
        self.sub_lane_visible = self.create_subscription(
            Bool, '/lane/center/visible', self.cb_lane_visible, 10)
        self.sub_curvature = self.create_subscription(
            Float32, '/lane/curvature', self.cb_curvature, 10)
        
        # Subscriber para exploration goals
        self.sub_exploration_goal = self.create_subscription(
            PoseStamped, self.exploration_goal_topic, self.cb_exploration_goal, 10)
        
        # Subscribers para detección de carril (forzar Nav2)
        self.sub_yellow_visible = self.create_subscription(
            Bool, self.yellow_visible_topic, self.cb_yellow_visible, 10)
        self.sub_edge_count = self.create_subscription(
            Float32, self.edge_count_topic, self.cb_edge_count, 10)
        self.sub_edge_position = self.create_subscription(
            Float32, self.edge_position_topic, self.cb_edge_position, 10)

        # ─── Pub ───
        self.pub_cmd = self.create_publisher(MotorCommands, '/qcar2_motor_speed_cmd', 10)
        
        # Publisher para enviar goals a Nav2
        self.pub_nav2_goal = self.create_publisher(PoseStamped, self.nav2_goal_topic, 10)

        # Debug pubs
        self.pub_hybrid_steer = self.create_publisher(Float32, '/hybrid/steering', 10)
        self.pub_hybrid_speed = self.create_publisher(Float32, '/hybrid/speed', 10)
        self.pub_hybrid_mode = self.create_publisher(Float32, '/hybrid/mode', 10)
        self.pub_detection_quality = self.create_publisher(Float32, '/hybrid/detection_quality', 10)
        # Detection quality values: 0=NONE, 1=PARTIAL_EDGE, 2=PARTIAL_YELLOW, 3=ACCEPTABLE, 4=GOOD

        # ─── Timer ───
        period = 1.0 / max(1.0, self.rate_hz)
        self.timer = self.create_timer(period, self.control_loop)

        # ─── Param callback ───
        self.add_on_set_parameters_callback(self._on_params_change)

        self.get_logger().info('HybridController v4 started (MotorCommands interface + Hybrid anti-minima)')
        self.get_logger().info(f'  Subscribes to {self.lane_cmd_topic} (MotorCommands from lane follower)')
        self.get_logger().info(f'  Subscribes to /cmd_vel_nav (Nav2 velocity)')
        self.get_logger().info(f'  Subscribes to {self.exploration_goal_topic} (exploration goals)')
        self.get_logger().info(f'  Subscribes to {self.yellow_visible_topic} (yellow line detection)')
        self.get_logger().info(f'  Subscribes to {self.edge_count_topic} (edge count)')
        self.get_logger().info(f'  Publishes to {self.nav2_goal_topic} (Nav2 goal)')
        self.get_logger().info(f'  max_angle={self.max_angle} rad')
        self.get_logger().info(f'  base_autonomous_speed={self.base_autonomous_speed} (fallback)')
        self.get_logger().info(f'  hybrid_use_max_speed={self.hybrid_use_max_speed} (anti-minima)')
        self.get_logger().info(f'  Publishes to /qcar2_motor_speed_cmd (MotorCommands)')
        self.get_logger().info(f'  MODE: LANE_ONLY until first exploration goal received')

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

    def cb_yellow_visible(self, msg: Bool):
        """Callback para visibilidad de línea amarilla."""
        self.yellow_visible = msg.data
        self._update_detection_state()

    def cb_edge_count(self, msg: Float32):
        """Callback para número de elementos detectados (0=nada, 1=parcial, 2=completo)."""
        self.edge_count = int(msg.data)
        self._update_detection_state()

    def cb_edge_position(self, msg: Float32):
        """Callback para posición normalizada del borde rojo (-1 a 1, 999=sin dato)."""
        self.edge_position = msg.data

    def _update_detection_state(self):
        """
        Actualiza el estado de detección y decide el modo de control.
        
        LÓGICA SIMPLIFICADA:
        - Si lane_visible=True (el nodo calculó un centro válido) → LANE_PID OK
        - Si lane_visible=False → NAV2_FORCED
        
        La calidad de detección se usa para logging y debug:
        - GOOD: amarilla + borde visible
        - ACCEPTABLE: solo borde visible pero lane_visible=True
        - PARTIAL_YELLOW: solo amarilla visible
        - PARTIAL_EDGE: solo borde pero lane_visible=False  
        - NONE: nada detectado
        """
        old_force = self.force_nav2
        old_quality = getattr(self, 'detection_quality', 'UNKNOWN')
        
        # REGLA PRINCIPAL: Si lane_visible=True, el nodo pudo calcular el centro → OK para PID
        # Esto cubre el caso de 2 líneas rojas sin amarilla
        if self.lane_visible:
            if self.yellow_visible and self.edge_count >= 1:
                self.detection_quality = 'GOOD'
            elif self.edge_count >= 1:
                self.detection_quality = 'ACCEPTABLE'  # Solo borde pero centro calculado
            elif self.yellow_visible:
                self.detection_quality = 'PARTIAL_YELLOW'
            else:
                self.detection_quality = 'ACCEPTABLE'  # Centro disponible por algún método
            self.force_nav2 = False
        else:
            # lane_visible=False → no hay centro calculado, forzar Nav2
            if self.yellow_visible:
                self.detection_quality = 'PARTIAL_YELLOW'
            elif self.edge_count >= 1:
                self.detection_quality = 'PARTIAL_EDGE'
            else:
                self.detection_quality = 'NONE'
            self.force_nav2 = True
        
        # Log cambios de estado
        if old_force != self.force_nav2 or old_quality != self.detection_quality:
            if self.force_nav2:
                self.get_logger().warn(
                    f'⚠️  NAV2_FORCED — Detection: {self.detection_quality} '
                    f'(lane_visible={self.lane_visible}, yellow={self.yellow_visible}, edges={self.edge_count})'
                )
            else:
                self.get_logger().info(
                    f'✅ LANE_PID — Detection: {self.detection_quality} '
                    f'(lane_visible={self.lane_visible}, yellow={self.yellow_visible}, edges={self.edge_count})'
                )

    def cb_exploration_goal(self, msg: PoseStamped):
        """
        Recibe goals de exploration_manager_node y los reenvía a Nav2.
        La primera vez que recibe un goal, activa el modo híbrido Nav2+lane.
        """
        self.current_goal = msg
        
        # Publicar el goal a Nav2
        self.pub_nav2_goal.publish(msg)
        
        if not self.has_received_goal:
            self.has_received_goal = True
            self.get_logger().info(
                '═══════════════════════════════════════════════════════════\n'
                '  🎯 FIRST EXPLORATION GOAL RECEIVED!\n'
                f'  Goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})\n'
                '  Switching from LANE_ONLY → HYBRID mode (Nav2 + Lane)\n'
                '═══════════════════════════════════════════════════════════'
            )
        else:
            self.get_logger().info(
                f'New exploration goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}) '
                f'→ forwarded to {self.nav2_goal_topic}'
            )

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

        # ══════════════════════════════════════════════════════════════════
        # MODO LANE_ONLY: antes de recibir el primer exploration goal
        # Usa directamente steering+speed del lane follower, ignora Nav2
        # CONDICIÓN MADRE: Sin goal → PID del lane follower exclusivamente
        # ══════════════════════════════════════════════════════════════════
        if not self.has_received_goal:
            # Sin goal aún → modo LANE_ONLY (solo lane follower)
            
            # Para usar LANE_PID necesitamos buena detección:
            # - lane_visible (centro calculado)
            # - No force_nav2 (al menos amarilla+1borde O 2 bordes)
            lane_ok = (dt_steering < self.lost_timeout and 
                       self.lane_visible and 
                       not self.force_nav2)
            
            if not lane_ok:
                # Lane perdido o detección insuficiente → parar y esperar
                self._publish_motor(0.0, 0.0)
                self._pub_debug(0.0, 0.0, MODE_STOPPED)
                self.prev_steering = 0.0
                
                # Log warning con throttle manual (cada 2 segundos)
                now = self.get_clock().now()
                if (now - self.last_lane_only_warn).nanoseconds > 2e9:
                    self.last_lane_only_warn = now
                    if self.force_nav2:
                        self.get_logger().warning(
                            f'🚫 LANE_ONLY: Detection poor ({self.detection_quality}), waiting for better detection...'
                        )
                    else:
                        self.get_logger().warning('🚫 LANE_ONLY: Lane lost, waiting...')
                return

            # Usar steering y speed directamente del lane follower (PID)
            steering_desired = clamp(self.lane_steering, -self.max_angle, self.max_angle)
            
            # Rate limiter
            dt = 1.0 / max(1.0, self.rate_hz)
            max_delta = self.max_steer_rate * dt
            delta = clamp(steering_desired - self.prev_steering, -max_delta, max_delta)
            steering = clamp(self.prev_steering + delta, -self.max_angle, self.max_angle)
            self.prev_steering = steering

            # Speed del lane follower o base_autonomous_speed
            if abs(self.lane_speed) > 0.001:
                speed = abs(self.lane_speed)
            else:
                speed = self.base_autonomous_speed

            # Curve slowdown
            curv_norm = clamp(abs(self.lane_curvature) * 1000.0, 0.0, 1.0)
            curve_factor = 1.0 - self.curve_slowdown_gain * curv_norm
            curve_factor = clamp(curve_factor, 0.3, 1.0)
            speed = speed * curve_factor
            speed = clamp(speed, self.min_speed, self.max_speed)

            self._publish_motor(steering, speed)
            self._pub_debug(steering, speed, MODE_LANE_ONLY)
            return

        # ══════════════════════════════════════════════════════════════════
        # MODO HÍBRIDO: después de recibir al menos un exploration goal
        # Depende de Nav2 para navegación + lane follower para steering fino
        # ══════════════════════════════════════════════════════════════════

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
        # 
        # IMPORTANTE: Cuando hay un goal activo, Nav2 tiene prioridad
        # para la dirección. Lane follower solo ayuda a mantenerse en carril.
        # ══════════════════════════════════════════════
        
        # ── NAV2_FORCED: detección de carril insuficiente → confiar 100% en Nav2 ──
        if self.force_nav2:
            mode = MODE_NAV2_FORCED
            steering_desired = clamp(
                self.nav2_forced_steer_gain * self.nav2_angular_z,
                -self.max_angle, self.max_angle
            )
            effective_rate = self.max_steer_rate
        
        elif self.in_nav2_turn:
            # ── NAV2_TURN: Nav2 controls steering (intersection/large turns) ──
            mode = MODE_NAV2_TURN
            steering_desired = clamp(
                self.turn_steer_gain * self.nav2_angular_z,
                -self.max_angle, self.max_angle
            )
            effective_rate = self.turn_max_steer_rate

        elif dt_steering < self.lost_timeout and self.lane_visible:
            # ── HYBRID STEERING: Mezcla ponderada de Nav2 + lane follower ──
            # Nav2 da la dirección principal hacia el goal
            # Lane follower ayuda a mantenerse dentro del carril
            mode = MODE_LANE_PID
            
            nav2_steer = self.turn_steer_gain * self.nav2_angular_z
            lane_steer = self.lane_steering
            
            # Mezcla ponderada: Nav2 tiene más peso para ir al goal
            steering_desired = (
                self.hybrid_nav2_steer_weight * nav2_steer +
                self.hybrid_lane_steer_weight * lane_steer
            )
            steering_desired = clamp(steering_desired, -self.max_angle, self.max_angle)
            effective_rate = self.max_steer_rate

        else:
            # ── FALLBACK: lane lost + no turn → use Nav2 angular.z ──
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
        
        # ══════════════════════════════════════════════
        # EDGE AVOIDANCE: Alejarse de la línea roja en NAV2_FORCED
        # 
        # Cuando Nav2 está controlando y hay una línea roja detectada:
        # - Si la línea está a la DERECHA → girar a la IZQUIERDA
        # - Si la línea está a la IZQUIERDA → girar a la DERECHA
        # Esto hace que el robot se aleje de la línea roja
        # ══════════════════════════════════════════════
        if self.edge_safety_enabled and self.force_nav2 and self.edge_position != 999.0:
            # edge_position: positivo = línea a la derecha, negativo = línea a la izquierda
            # steering: positivo = girar a la izquierda, negativo = girar a la derecha
            
            edge_is_close = abs(self.edge_position) > self.edge_safety_threshold
            
            if edge_is_close:
                if self.edge_position > 0:
                    # Línea roja está a la DERECHA → FORZAR giro a la IZQUIERDA (steering positivo)
                    steering = self.edge_safety_steer_limit
                    self.get_logger().debug(
                        f'🛑 Edge avoidance: line RIGHT at {self.edge_position:.2f}, '
                        f'steering LEFT to {steering:.3f} rad'
                    )
                else:
                    # Línea roja está a la IZQUIERDA → FORZAR giro a la DERECHA (steering negativo)
                    steering = -self.edge_safety_steer_limit
                    self.get_logger().debug(
                        f'🛑 Edge avoidance: line LEFT at {self.edge_position:.2f}, '
                        f'steering RIGHT to {steering:.3f} rad'
                    )
        
        self.prev_steering = steering

        # ══════════════════════════════════════════════
        # SPEED decision - HYBRID INTELLIGENCE
        # 
        # Estrategia para evitar mínimos locales en Ackermann:
        # - NAV2_FORCED: Solo Nav2 (sin detección de carril)
        # - LANE_PID: Mezcla inteligente de Nav2 + lane follower
        #   → Nav2 da la dirección hacia el goal
        #   → Lane follower da velocidad constante para superar mínimos locales
        # ══════════════════════════════════════════════
        
        # ── NAV2_FORCED: Sin detección suficiente → confiar 100% en Nav2 ──
        if self.force_nav2:
            if abs(self.nav2_linear_x) > 0.001:
                base_speed = abs(self.nav2_linear_x) * self.nav2_forced_speed_scale
            else:
                base_speed = 0.0  # Sin Nav2 velocity cuando está forzado → parar
        
        # ── HYBRID MODE: Combinar Nav2 + lane follower para evitar mínimos locales ──
        elif self.has_received_goal and not self.force_nav2:
            # ESTRATEGIA ANTI MÍNIMOS LOCALES:
            # El robot Ackermann se atasca cuando Nav2 reduce velocidad cerca de obstáculos
            # o cuando no encuentra camino directo. Solución: usar velocidad del lane follower
            # como "empuje" mientras Nav2 proporciona la dirección.
            
            nav2_has_velocity = abs(self.nav2_linear_x) > 0.001
            lane_has_velocity = abs(self.lane_speed) > 0.001 and self.lane_visible
            
            if nav2_has_velocity and (lane_has_velocity or self.lane_visible):
                # MODO HÍBRIDO ACTIVO: Mezclar velocidades
                # Nav2 velocity (scaled) como mínimo, lane velocity como boost
                nav2_speed = abs(self.nav2_linear_x) * self.nav2_speed_scale
                
                if lane_has_velocity:
                    lane_speed_val = abs(self.lane_speed)
                else:
                    lane_speed_val = self.base_autonomous_speed
                
                if self.hybrid_use_max_speed:
                    # Tomar el MÁXIMO de ambas velocidades para superar mínimos locales
                    # Esto asegura que el robot siga avanzando incluso si Nav2 reduce velocidad
                    base_speed = max(nav2_speed, lane_speed_val * self.hybrid_lane_speed_factor)
                else:
                    # Usar solo Nav2 speed (comportamiento conservador)
                    base_speed = nav2_speed
                
                # Pero no exceder el máximo permitido
                base_speed = min(base_speed, self.max_speed)
                
            elif nav2_has_velocity:
                # Solo Nav2 tiene velocidad
                base_speed = abs(self.nav2_linear_x) * self.nav2_speed_scale
            elif lane_has_velocity:
                # Solo lane follower tiene velocidad
                base_speed = abs(self.lane_speed)
            elif self.lane_visible:
                base_speed = self.base_autonomous_speed
            else:
                base_speed = 0.0
        
        # ── LANE_ONLY MODE: Sin goal recibido → solo lane follower ──
        else:
            if abs(self.lane_speed) > 0.001 and self.lane_visible:
                base_speed = abs(self.lane_speed)
            elif self.lane_visible:
                base_speed = self.base_autonomous_speed
            else:
                base_speed = 0.0

        # Curve/error slowdown (reducir en curvas pronunciadas)
        curv_norm = clamp(abs(self.lane_curvature) * 1000.0, 0.0, 1.0)
        curve_factor = 1.0 - self.curve_slowdown_gain * curv_norm
        curve_factor = clamp(curve_factor, 0.3, 1.0)
        
        speed = base_speed * curve_factor

        # Extra slowdown durante giros Nav2 (intersecciones)
        if self.in_nav2_turn:
            speed *= self.turn_speed_scale

        # Preserve direction from Nav2
        if self.nav2_linear_x < 0:
            speed = -speed

        speed = clamp(speed, -self.max_speed, self.max_speed)
        
        # Velocidad mínima para evitar que el robot se detenga en mínimos locales
        # Solo aplicar si hay un goal activo y Nav2 quiere moverse
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
        
        # Publish detection quality as numeric value for monitoring
        # 0=NONE, 1=PARTIAL_EDGE, 2=PARTIAL_YELLOW, 3=ACCEPTABLE, 4=GOOD
        quality_map = {
            'NONE': 0.0,
            'PARTIAL_EDGE': 1.0,
            'PARTIAL_YELLOW': 2.0,
            'ACCEPTABLE': 3.0,
            'GOOD': 4.0
        }
        quality_val = quality_map.get(self.detection_quality, 0.0)
        self.pub_detection_quality.publish(Float32(data=quality_val))


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
