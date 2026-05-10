#!/usr/bin/env python3
"""
PID Tuner Node (ROS2) — QCar2

Nodo sin GUI que expone TODOS los parámetros del
yellow_line_follower_controller como parámetros ROS2 propios.
Se tunean desde rqt_reconfigure y los cambios se propagan
automáticamente al controlador vía set_parameters service.
"""
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.msg import SetParametersResult


# ---------------------------------------------------------------
# Parámetros a tunear.
# (nombre_param, default, tipo: 'double' | 'bool')
# ---------------------------------------------------------------
TUNABLE_PARAMS = [
    # PID
    ('kp',                  0.42105263157894735, 'double'),
    ('ki',                  0.031578947368421054, 'double'),
    ('kd',                  0.16842105263157894, 'double'),
    ('integral_limit',      0.8,                'double'),
    # Límites / velocidad
    ('max_angle',           0.45,               'double'),
    ('base_speed',          0.35,               'double'),
    ('min_speed',           0.10,               'double'),
    ('max_speed',           0.20,               'double'),
    ('slowdown_gain',       0.65,               'double'),
    ('curve_slowdown_gain', 0.4,                'double'),
    # Suavizado / derivada
    ('max_steer_rate',      1.5,                'double'),
    ('visible_hold_sec',    0.20,               'double'),
    ('derivative_limit',    8.0,                'double'),
    # Kalman
    ('use_kalman',          True,               'bool'),
    ('kalman_q',            0.02,               'double'),
    ('kalman_r',            0.08,               'double'),
]


class PIDTunerNode(Node):
    """
    Nodo ROS2 que actúa como proxy de parámetros.
    Cada parámetro tunable se declara aquí y cuando cambia
    (vía rqt_reconfigure) se propaga al controlador objetivo.
    """

    def __init__(self):
        super().__init__('pid_tuner')

        # --- Config ---
        self.declare_parameter('target_node', 'yellow_line_follower_controller')
        self.declare_parameter('push_rate_hz', 5.0)

        self.target_node_name = str(self.get_parameter('target_node').value)
        self.push_rate_hz = float(self.get_parameter('push_rate_hz').value)

        # Declarar cada parámetro tunable como parámetro propio
        for name, default, ptype in TUNABLE_PARAMS:
            self.declare_parameter(name, default)

        # Leer valores iniciales
        self._values = {}
        for name, _default, ptype in TUNABLE_PARAMS:
            v = self.get_parameter(name).value
            self._values[name] = bool(v) if ptype == 'bool' else float(v)

        # Cliente set_parameters del nodo objetivo
        self.set_srv_name = f'/{self.target_node_name}/set_parameters'
        self.cli_set = self.create_client(SetParameters, self.set_srv_name)

        # Último estado enviado (para detectar cambios)
        self._last_sent = dict(self._values)

        # Callback: cuando rqt cambia un parámetro aquí, lo marcamos
        self.add_on_set_parameters_callback(self._on_param_change)

        # Timer periódico para propagar cambios
        period = 1.0 / max(0.5, self.push_rate_hz)
        self._timer = self.create_timer(period, self._push_if_changed)

        # Log
        self.get_logger().info(
            f"PID Tuner started → target: {self.target_node_name}")
        self.get_logger().info(
            f"  Adjust params with: ros2 run rqt_reconfigure rqt_reconfigure")

    # ----------------------------------------------------------------
    # Callback: detecta cambios hechos desde rqt_reconfigure
    # ----------------------------------------------------------------
    def _on_param_change(self, params):
        for p in params:
            if p.name in self._values:
                ptype = next(
                    (t for n, _d, t in TUNABLE_PARAMS if n == p.name), None)
                if ptype == 'bool':
                    self._values[p.name] = bool(p.value)
                elif ptype == 'double':
                    self._values[p.name] = float(p.value)
        return SetParametersResult(successful=True)

    # ----------------------------------------------------------------
    # Propagar cambios al controlador
    # ----------------------------------------------------------------
    def _push_if_changed(self):
        # Detectar qué cambió
        changed = {k: v for k, v in self._values.items()
                   if self._last_sent.get(k) != v}
        if not changed:
            return

        if not self.cli_set.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn(
                f"Servicio {self.set_srv_name} no disponible", throttle_duration_sec=5.0)
            return

        req = SetParameters.Request()
        req.parameters = [self._make_param(k, v) for k, v in changed.items()]

        future = self.cli_set.call_async(req)
        future.add_done_callback(
            lambda fut: self._on_push_result(fut, changed))

    def _on_push_result(self, future, changed):
        try:
            res = future.result()
            if all(r.successful for r in res.results):
                self._last_sent.update(changed)
                names = ', '.join(f"{k}={v}" for k, v in changed.items())
                self.get_logger().info(f"Propagado → {names}")
            else:
                self.get_logger().warn("Algunos parámetros fallaron")
        except Exception as e:
            self.get_logger().error(f"Error propagando params: {e}")

    @staticmethod
    def _make_param(name: str, value):
        p = Parameter()
        p.name = name
        pv = ParameterValue()
        if isinstance(value, bool):
            pv.type = ParameterType.PARAMETER_BOOL
            pv.bool_value = bool(value)
        else:
            pv.type = ParameterType.PARAMETER_DOUBLE
            pv.double_value = float(value)
        p.value = pv
        return p


def main(args=None):
    rclpy.init(args=args)
    node = PIDTunerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
