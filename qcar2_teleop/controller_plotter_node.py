#!/usr/bin/env python3
import time
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from qcar2_interfaces.msg import MotorCommands

import matplotlib.pyplot as plt
import matplotlib.animation as animation


class ControllerPlotter(Node):
    """
    Ventana matplotlib con 2 gráficas (una sola ventana):
      1) Error vs setpoint (0) + "control real" (steering cmd)
      2) Steering y Speed (throttle) vs tiempo

    Nota: el "control signal" más real que puedes ver sin tocar el controlador
    es el steering command publicado en /qcar2_motor_speed_cmd.
    """

    def __init__(self):
        super().__init__('controller_plotter')

        self.declare_parameter('error_topic', '/lane/center/error')
        self.declare_parameter('visible_topic', '/lane/center/visible')
        self.declare_parameter('cmd_topic', '/qcar2_motor_speed_cmd')

        self.declare_parameter('history_sec', 12.0)
        self.declare_parameter('update_hz', 20.0)

        self.error_topic = self.get_parameter('error_topic').value
        self.visible_topic = self.get_parameter('visible_topic').value
        self.cmd_topic = self.get_parameter('cmd_topic').value

        self.history_sec = float(self.get_parameter('history_sec').value)
        self.update_hz = float(self.get_parameter('update_hz').value)

        self.maxlen = int(self.history_sec * self.update_hz) + 5

        # Buffers
        self.t = deque(maxlen=self.maxlen)
        self.err = deque(maxlen=self.maxlen)
        self.vis = deque(maxlen=self.maxlen)
        self.steer = deque(maxlen=self.maxlen)
        self.speed = deque(maxlen=self.maxlen)

        # last values
        self.last_err = 0.0
        self.last_vis = False
        self.last_steer = 0.0
        self.last_speed = 0.0

        self.sub_e = self.create_subscription(Float32, self.error_topic, self.cb_error, 10)
        self.sub_v = self.create_subscription(Bool, self.visible_topic, self.cb_visible, 10)
        self.sub_c = self.create_subscription(MotorCommands, self.cmd_topic, self.cb_cmd, 10)

        self.start_time = time.time()

        # Matplotlib setup (una sola ventana, 2 plots)
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, sharex=True)
        self.fig.canvas.manager.set_window_title("Controller Live Plots")

        # Plot 1: error + setpoint + steering (control)
        (self.l_err,) = self.ax1.plot([], [], label="error (lane/center/error)")
        (self.l_sp,) = self.ax1.plot([], [], "--", label="setpoint=0")
        (self.l_u,) = self.ax1.plot([], [], label="control ≈ steering_cmd")

        self.ax1.set_ylabel("Error / Steering")
        self.ax1.grid(True)
        self.ax1.legend(loc="upper right")

        # Plot 2: steering + speed
        (self.l_s,) = self.ax2.plot([], [], label="steering_cmd")
        (self.l_v,) = self.ax2.plot([], [], label="speed_cmd (throttle)")
        self.ax2.set_ylabel("Cmd")
        self.ax2.set_xlabel("Time [s]")
        self.ax2.grid(True)
        self.ax2.legend(loc="upper right")

        self.timer = self.create_timer(1.0 / self.update_hz, self.sample_tick)

        # animation
        self.ani = animation.FuncAnimation(self.fig, self._animate, interval=int(1000 / self.update_hz), blit=False)

        self.get_logger().info("ControllerPlotter started")

    def cb_error(self, msg: Float32):
        self.last_err = float(msg.data)

    def cb_visible(self, msg: Bool):
        self.last_vis = bool(msg.data)

    def cb_cmd(self, msg: MotorCommands):
        # OJO: tu controlador publica values = [steering, speed]
        if len(msg.values) >= 2:
            self.last_steer = float(msg.values[0])
            self.last_speed = float(msg.values[1])

    def sample_tick(self):
        now = time.time() - self.start_time
        self.t.append(now)
        self.err.append(self.last_err)
        self.vis.append(1.0 if self.last_vis else 0.0)
        self.steer.append(self.last_steer)
        self.speed.append(self.last_speed)

    def _animate(self, _):
        if len(self.t) < 5:
            return

        t = list(self.t)
        err = list(self.err)
        steer = list(self.steer)
        speed = list(self.speed)

        # Plot 1
        self.l_err.set_data(t, err)
        self.l_sp.set_data(t, [0.0] * len(t))
        self.l_u.set_data(t, steer)

        # Auto limits
        self.ax1.set_xlim(max(0.0, t[-1] - self.history_sec), t[-1] + 0.01)
        y1min = min(min(err), min(steer), -0.1)
        y1max = max(max(err), max(steer), 0.1)
        self.ax1.set_ylim(y1min - 0.05, y1max + 0.05)

        # Plot 2
        self.l_s.set_data(t, steer)
        self.l_v.set_data(t, speed)
        y2min = min(min(steer), min(speed), -0.1)
        y2max = max(max(steer), max(speed), 0.1)
        self.ax2.set_ylim(y2min - 0.05, y2max + 0.05)

        return

    def show(self):
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = ControllerPlotter()
    try:
        # Matplotlib bloquea; usamos spin en un thread implícito? aquí basta porque callbacks corren con timers.
        # Para que ROS procese callbacks mientras plt.show corre, hacemos spin en background:
        import threading
        spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        spin_thread.start()

        node.show()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
