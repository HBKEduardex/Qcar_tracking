#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from qcar2_interfaces.msg import MotorCommands
from rcl_interfaces.msg import SetParametersResult


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class ScalarKalman1D:
    """
    Kalman 1D (estado = error lateral).
    Modelo: x_k = x_{k-1}  (random walk)
    Medición: z_k = x_k + v
    """
    def __init__(self, x0=0.0, p0=1.0, q=0.02, r=0.08):
        self.x = float(x0)
        self.p = float(p0)
        self.q = float(q)   # process noise
        self.r = float(r)   # measurement noise

    def reset(self, x0=0.0, p0=1.0):
        self.x = float(x0)
        self.p = float(p0)

    def update(self, z: float) -> float:
        # Predict
        self.p = self.p + self.q
        # Update
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (z - self.x)
        self.p = (1.0 - k) * self.p
        return self.x


class YellowLineFollowerController(Node):
    """
    Control PID para alinear el LANE CENTER con el centro de la imagen:
      - setpoint = 0
      - error = /lane/center/error (Float32, [-1,+1])
      - curvature = /lane/curvature (Float32) → frena más en curvas

    Publica MotorCommands (manteniendo tu convención):
      msg.motor_names = ['motor_throttle', 'steering_angle']
      msg.values      = [steering_angle, motor_throttle]

    Además publica debug (para el nodo de gráficas):
      /controller/error_raw   (Float32)
      /controller/error_filt  (Float32)
      /controller/u_raw       (Float32)
      /controller/steering_cmd(Float32)
      /controller/speed_cmd   (Float32)
      /controller/curvature   (Float32)
    """

    def __init__(self):
        super().__init__('yellow_line_follower_controller')

        # Topics
        self.declare_parameter('error_topic', '/lane/center/error')
        self.declare_parameter('visible_topic', '/lane/center/visible')
        self.declare_parameter('curvature_topic', '/lane/curvature')
        self.declare_parameter('cmd_topic', '/qcar2_motor_speed_cmd')

        # PID
        self.declare_parameter('kp', 0.42105263157894735)
        self.declare_parameter('ki', 0.031578947368421054)
        self.declare_parameter('kd', 0.16842105263157894)
        self.declare_parameter('integral_limit', 0.8)

        # Limits / speed
        self.declare_parameter('max_angle', 0.45)     # rad
        self.declare_parameter('base_speed', 0.35)
        self.declare_parameter('min_speed', 0.15)
        self.declare_parameter('max_speed', 0.30)
        self.declare_parameter('slowdown_gain', 0.65)  # baja speed con |error|
        self.declare_parameter('curve_slowdown_gain', 0.4)  # baja speed extra en curvas

        # Safety / rate
        self.declare_parameter('lost_timeout', 0.35)
        self.declare_parameter('rate_hz', 20.0)

        # Suavidad / curvas
        self.declare_parameter('max_steer_rate', 1.5)       # rad/s
        self.declare_parameter('visible_hold_sec', 0.20)    # hold si visible cae poco
        self.declare_parameter('derivative_limit', 8.0)     # limita derivada

        # Kalman
        self.declare_parameter('use_kalman', True)
        self.declare_parameter('kalman_q', 0.02)   # sube si responde lento
        self.declare_parameter('kalman_r', 0.08)   # sube si vibra
        self.declare_parameter('kalman_p0', 1.0)

        # Debug pubs
        self.declare_parameter('publish_debug', True)

        # Load params
        self.error_topic = self.get_parameter('error_topic').value
        self.visible_topic = self.get_parameter('visible_topic').value
        self.curvature_topic = self.get_parameter('curvature_topic').value
        self.cmd_topic = self.get_parameter('cmd_topic').value

        self.kp = float(self.get_parameter('kp').value)
        self.ki = float(self.get_parameter('ki').value)
        self.kd = float(self.get_parameter('kd').value)
        self.integral_limit = float(self.get_parameter('integral_limit').value)

        self.max_angle = float(self.get_parameter('max_angle').value)
        self.base_speed = float(self.get_parameter('base_speed').value)
        self.min_speed = float(self.get_parameter('min_speed').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.slowdown_gain = float(self.get_parameter('slowdown_gain').value)
        self.curve_slowdown_gain = float(self.get_parameter('curve_slowdown_gain').value)

        self.lost_timeout = float(self.get_parameter('lost_timeout').value)
        self.rate_hz = float(self.get_parameter('rate_hz').value)

        self.max_steer_rate = float(self.get_parameter('max_steer_rate').value)
        self.visible_hold_sec = float(self.get_parameter('visible_hold_sec').value)
        self.derivative_limit = float(self.get_parameter('derivative_limit').value)

        self.use_kalman = bool(self.get_parameter('use_kalman').value)
        self.kalman_q = float(self.get_parameter('kalman_q').value)
        self.kalman_r = float(self.get_parameter('kalman_r').value)
        self.kalman_p0 = float(self.get_parameter('kalman_p0').value)

        self.publish_debug = bool(self.get_parameter('publish_debug').value)

        # State
        self.last_error = 0.0
        self.last_curvature = 0.0
        self.visible = False
        self.last_msg_time = self.get_clock().now()
        self.last_visible_true_time = self.get_clock().now()

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = self.get_clock().now()
        self.first = True

        self.prev_steering = 0.0

        # Kalman filter
        self.kf = ScalarKalman1D(x0=0.0, p0=self.kalman_p0, q=self.kalman_q, r=self.kalman_r)

        # Subs/Pub
        self.sub_err = self.create_subscription(Float32, self.error_topic, self.cb_error, 10)
        self.sub_vis = self.create_subscription(Bool, self.visible_topic, self.cb_visible, 10)
        self.sub_curv = self.create_subscription(Float32, self.curvature_topic, self.cb_curvature, 10)
        self.pub_cmd = self.create_publisher(MotorCommands, self.cmd_topic, 10)

        # Debug publishers
        if self.publish_debug:
            self.pub_error_raw = self.create_publisher(Float32, '/controller/error_raw', 10)
            self.pub_error_filt = self.create_publisher(Float32, '/controller/error_filt', 10)
            self.pub_u_raw = self.create_publisher(Float32, '/controller/u_raw', 10)
            self.pub_steer = self.create_publisher(Float32, '/controller/steering_cmd', 10)
            self.pub_speed = self.create_publisher(Float32, '/controller/speed_cmd', 10)
            self.pub_curvature = self.create_publisher(Float32, '/controller/curvature', 10)

        # ✅ CLAVE: permite que el PID Tuner cambie params en vivo
        self.add_on_set_parameters_callback(self._on_params_change)

        self.timer = self.create_timer(1.0 / self.rate_hz, self.control_loop)

        self.get_logger().info("YellowLineFollowerController (PID + Kalman + curve-aware) started")
        self.get_logger().info(f"  error_topic    : {self.error_topic}")
        self.get_logger().info(f"  visible_topic  : {self.visible_topic}")
        self.get_logger().info(f"  curvature_topic: {self.curvature_topic}")
        self.get_logger().info(f"  cmd_topic      : {self.cmd_topic}")
        self.get_logger().info(f"  curve_slowdown_gain={self.curve_slowdown_gain}")
        self.get_logger().info(f"  use_kalman={self.use_kalman} (q={self.kalman_q}, r={self.kalman_r})")

    # ============ Live parameter updates ============
    def _on_params_change(self, params):
        for p in params:
            if p.name == 'kp':
                self.kp = float(p.value)
            elif p.name == 'ki':
                self.ki = float(p.value)
            elif p.name == 'kd':
                self.kd = float(p.value)
            elif p.name == 'integral_limit':
                self.integral_limit = float(p.value)
            elif p.name == 'max_angle':
                self.max_angle = float(p.value)
            elif p.name == 'base_speed':
                self.base_speed = float(p.value)
            elif p.name == 'min_speed':
                self.min_speed = float(p.value)
            elif p.name == 'max_speed':
                self.max_speed = float(p.value)
            elif p.name == 'curve_slowdown_gain':
                self.curve_slowdown_gain = float(p.value)
            elif p.name == 'slowdown_gain':
                self.slowdown_gain = float(p.value)
            elif p.name == 'max_steer_rate':
                self.max_steer_rate = float(p.value)
            elif p.name == 'visible_hold_sec':
                self.visible_hold_sec = float(p.value)
            elif p.name == 'derivative_limit':
                self.derivative_limit = float(p.value)
            elif p.name == 'use_kalman':
                self.use_kalman = bool(p.value)
            elif p.name == 'kalman_q':
                self.kalman_q = float(p.value)
                self.kf.q = self.kalman_q
            elif p.name == 'kalman_r':
                self.kalman_r = float(p.value)
                self.kf.r = self.kalman_r
        return SetParametersResult(successful=True)

    def cb_error(self, msg: Float32):
        self.last_error = float(msg.data)
        self.last_msg_time = self.get_clock().now()

    def cb_visible(self, msg: Bool):
        self.visible = bool(msg.data)
        self.last_msg_time = self.get_clock().now()
        if self.visible:
            self.last_visible_true_time = self.get_clock().now()

    def cb_curvature(self, msg: Float32):
        self.last_curvature = float(msg.data)

    def publish_motorcommands(self, steering_angle: float, motor_throttle: float):
        msg = MotorCommands()
        # Mantengo tu convención para no romper tu sistema actual:
        msg.motor_names = ['motor_throttle', 'steering_angle']
        msg.values = [float(steering_angle), float(motor_throttle)]
        self.pub_cmd.publish(msg)

    def _pub_debug(self, z, ef, u_raw, steering, speed):
        if not self.publish_debug:
            return
        m = Float32()

        m.data = float(z)
        self.pub_error_raw.publish(m)

        m.data = float(ef)
        self.pub_error_filt.publish(m)

        m.data = float(u_raw)
        self.pub_u_raw.publish(m)

        m.data = float(steering)
        self.pub_steer.publish(m)

        m.data = float(speed)
        self.pub_speed.publish(m)

        m.data = float(self.last_curvature)
        self.pub_curvature.publish(m)

    def _reset_controller_state(self):
        self.integral = 0.0
        self.first = True
        self.prev_steering = 0.0
        self.prev_error = 0.0
        self.prev_time = self.get_clock().now()
        self.kf.reset(0.0, self.kalman_p0)

    def control_loop(self):
        now = self.get_clock().now()

        # Safety: si no llegan msgs recientes, frena
        dt_lost = (now - self.last_msg_time).nanoseconds * 1e-9
        if dt_lost > self.lost_timeout:
            self.publish_motorcommands(0.0, 0.0)
            self._reset_controller_state()
            return

        # Hold corto si visible cae por microcortes
        dt_since_vis = (now - self.last_visible_true_time).nanoseconds * 1e-9
        effective_visible = self.visible or (dt_since_vis <= self.visible_hold_sec)

        if not effective_visible:
            self.publish_motorcommands(0.0, 0.0)
            self._reset_controller_state()
            return

        # dt
        dt = (now - self.prev_time).nanoseconds * 1e-9
        if dt <= 1e-6:
            dt = 1.0 / self.rate_hz
        self.prev_time = now

        # Medición (error)
        z = clamp(self.last_error, -1.0, 1.0)

        # Kalman (suaviza)
        if self.use_kalman:
            ef = float(self.kf.update(z))
            ef = clamp(ef, -1.0, 1.0)
        else:
            ef = z

        # PID
        if self.first:
            self.prev_error = ef
            self.first = False

        derivative = (ef - self.prev_error) / dt
        derivative = clamp(derivative, -abs(self.derivative_limit), abs(self.derivative_limit))
        self.prev_error = ef

        # u sin integral (para anti-windup y debug)
        u_no_i = (self.kp * ef) + (self.kd * derivative)

        steering_no_i = clamp(-u_no_i, -self.max_angle, self.max_angle)
        would_saturate = abs(steering_no_i) >= (self.max_angle - 1e-6)

        if self.ki != 0.0 and not would_saturate:
            self.integral += ef * dt
            self.integral = clamp(self.integral, -self.integral_limit, self.integral_limit)

        u = u_no_i + (self.ki * self.integral)

        steering_desired = clamp(-u, -self.max_angle, self.max_angle)

        # Rate limiter
        max_delta = abs(self.max_steer_rate) * dt
        delta = clamp(steering_desired - self.prev_steering, -max_delta, max_delta)
        steering = clamp(self.prev_steering + delta, -self.max_angle, self.max_angle)
        self.prev_steering = steering

        # Speed adaptativa (+ curvatura)
        curv_norm = clamp(abs(self.last_curvature) * 1000.0, 0.0, 1.0)  # normalizar
        curve_factor = 1.0 - self.curve_slowdown_gain * curv_norm
        curve_factor = clamp(curve_factor, 0.3, 1.0)
        speed = self.base_speed * curve_factor * (1.0 - self.slowdown_gain * abs(ef))
        speed = clamp(speed, self.min_speed, self.max_speed)

        # Debug pubs
        self._pub_debug(z=z, ef=ef, u_raw=u, steering=steering, speed=speed)

        self.publish_motorcommands(steering, speed)


def main(args=None):
    rclpy.init(args=args)
    node = YellowLineFollowerController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop motors before shutting down
        node.publish_motorcommands(0.0, 0.0)
        node.get_logger().info("Motors stopped. Shutting down.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
