#!/usr/bin/env python3
import time
import math
from collections import deque
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

# Se usa matplotlib.use('TkAgg') para mejor compatibilidad en hilos
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class PidMonitorPlotter(Node):
    """
    Monitor Externo Pasivo:
      - 2 Subplots (Error/U, Steering).
      - Congela la gráfica (freeze) cuando no está en PID Mode.
      - 100% pasivo: no publica, solo se suscribe.
    """

    def __init__(self):
        super().__init__('pid_monitor_plotter')

        # ─── Parámetros Configurables ───
        # Por defecto mapean a los tópicos detectados en el paquete qcar2_teleop
        self.declare_parameter('topic_error', '/controller/error_filt')
        self.declare_parameter('topic_pid_u', '/controller/u_raw')
        self.declare_parameter('topic_steering', '/hybrid/steering')
        self.declare_parameter('topic_mode', '/hybrid/mode')
        
        # Valor del modo híbrido que significa "PID ACTIVADO"
        # mode=1.0 es PID en hybrid_switch_controller_node.py
        self.declare_parameter('pid_mode_value', 1.0)
        
        self.declare_parameter('buffer_size', 500)
        self.declare_parameter('update_hz', 20.0)

        # ─── Obtener Valores ───
        topic_err = self.get_parameter('topic_error').value
        topic_u = self.get_parameter('topic_pid_u').value
        topic_steer = self.get_parameter('topic_steering').value
        topic_mode = self.get_parameter('topic_mode').value
        
        self.pid_mode_value = float(self.get_parameter('pid_mode_value').value)
        self.buffer_size = int(self.get_parameter('buffer_size').value)
        self.update_hz = float(self.get_parameter('update_hz').value)

        # ─── Estado local ───
        self.last_err = 0.0
        self.last_u = 0.0
        self.last_steer = 0.0
        
        self.is_pid_mode = False
        self.current_mode_val = -999.0

        # Buffers deslizantes para la gráfica
        self.t_data = deque(maxlen=self.buffer_size)
        self.err_data = deque(maxlen=self.buffer_size)
        self.u_data = deque(maxlen=self.buffer_size)
        self.steer_data = deque(maxlen=self.buffer_size)

        # ─── Suscriptores ───
        # Con base en la arquitectura actual, todo se publica como Float32
        self.sub_err = self.create_subscription(Float32, topic_err, self.cb_err, 10)
        self.sub_u = self.create_subscription(Float32, topic_u, self.cb_u, 10)
        self.sub_steer = self.create_subscription(Float32, topic_steer, self.cb_steer, 10)
        self.sub_mode = self.create_subscription(Float32, topic_mode, self.cb_mode, 10)

        self.start_time = time.time()

        # ─── Configuración de Matplotlib ───
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
        self.fig.canvas.manager.set_window_title("External PID Monitor - Hybrid")

        # Ajuste visual
        self.fig.patch.set_facecolor('#F0F0F0')

        # Plot 1: Error vs U
        (self.line_err,) = self.ax1.plot([], [], 'r-', linewidth=2, label='Error (Cross-Track)')
        (self.line_u,)   = self.ax1.plot([], [], 'b-', linewidth=2, label='PID Control (u)')
        (self.line_zero_err,) = self.ax1.plot([], [], 'k--', linewidth=1.5, label='0 (Reference)')
        self.ax1.set_ylabel("Magnitude")
        self.ax1.set_title("PID Reaction Analyzer")
        self.ax1.legend(loc="upper left")
        self.ax1.grid(True, linestyle='--', alpha=0.7)

        # Texto del MODO (Subplot 1)
        self.mode_text_1 = self.ax1.text(0.5, 0.9, 'WAITING FOR STATE...', 
                                         transform=self.ax1.transAxes,
                                         ha='center', va='center',
                                         fontsize=12, fontweight='bold',
                                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # Plot 2: Steering vs tiempo
        (self.line_steer,) = self.ax2.plot([], [], 'g-', linewidth=2, label='Steering Angle')
        (self.line_zero,)  = self.ax2.plot([], [], 'k--', linewidth=1.5, label='0 (Straight)')
        self.ax2.set_ylabel("Angle (rad)")
        self.ax2.set_xlabel("Relative Time (s)")
        self.ax2.legend(loc="upper left")
        self.ax2.grid(True, linestyle='--', alpha=0.7)

        self.get_logger().info("==========================================")
        self.get_logger().info(" EXTERNAL PID MONITOR STARTED (PASSIVE)")
        self.get_logger().info(f" -> Error Topic: {topic_err}")
        self.get_logger().info(f" -> Control(u) Topic: {topic_u}")
        self.get_logger().info(f" -> Steering Topic: {topic_steer}")
        self.get_logger().info(f" -> Mode Topic: {topic_mode} (Target Val = {self.pid_mode_value})")
        self.get_logger().info("==========================================")

        # Timer de ROS para recolectar las muestras
        self.create_timer(1.0 / self.update_hz, self.sample_data)

        # Animación de Matplotlib para actualizar la GUI
        self.ani = animation.FuncAnimation(self.fig, self.update_plot,
                                           interval=int(1000.0 / self.update_hz),
                                           blit=False, cache_frame_data=False)

    # ─── Callbacks de Suscripción ───
    def cb_err(self, msg: Float32):
        self.last_err = float(msg.data)

    def cb_u(self, msg: Float32):
        self.last_u = float(msg.data)

    def cb_steer(self, msg: Float32):
        self.last_steer = float(msg.data)

    def cb_mode(self, msg: Float32):
        val = float(msg.data)
        self.current_mode_val = val
        # Comparación robusta
        self.is_pid_mode = (abs(val - self.pid_mode_value) < 1e-3)

    # ─── Recolección de Datos ───
    def sample_data(self):
        """
        Guarda los datos en el buffer SOLO si el PID está activado.
        Si está en Nav2, no agregamos datos, haciendo un 'freeze'.
        """
        if not self.is_pid_mode:
            return
            
        now = time.time() - self.start_time
        self.t_data.append(now)
        self.err_data.append(self.last_err)
        self.u_data.append(self.last_u)
        self.steer_data.append(self.last_steer)

    # ─── Refresco de Matplotlib ───
    def update_plot(self, frame):
        # 1. Actualizar Textos y Colores de Estado siempre
        if self.current_mode_val == -999.0:
            txt = "WAITING FOR MODE TOPIC"
            color = '#808080' # Gris
        elif self.is_pid_mode:
            txt = "PID ACTIVE (Live Updating...)"
            color = '#2ca02c' # Verde
        elif self.current_mode_val == -1.0:
            txt = "STOPPED (No Mission)"
            color = '#d62728' # Rojo
        else:
            txt = "NAV2 ACTIVE (Plot Frozen)"
            color = '#ff7f0e' # Naranja / Amarillo oscuro
            
        self.mode_text_1.set_text(txt)
        self.mode_text_1.set_color(color)
        self.mode_text_1.set_alpha(1.0)
        self.ax1.set_facecolor('#ffffff' if self.is_pid_mode else '#fafafa')

        # 2. Refrescar líneas solo si hay datos en el buffer
        if not self.t_data:
            return

        t = list(self.t_data)
        e = list(self.err_data)
        u = list(self.u_data)
        s = list(self.steer_data)

        self.line_err.set_data(t, e)
        self.line_u.set_data(t, u)
        self.line_zero_err.set_data(t, [0.0] * len(t))
        self.line_steer.set_data(t, s)
        self.line_zero.set_data(t, [0.0] * len(t))

        # 3. Ajuste de Límites X (Ventana deslizante)
        t_max = t[-1]
        t_min = t[0]
        span = max(5.0, t_max - t_min)
        
        self.ax1.set_xlim(t_max - span, t_max + (span * 0.05))
        self.ax2.set_xlim(t_max - span, t_max + (span * 0.05))

        # 4. Ajuste de Límites Y dinámico (Subplot 1)
        ymin1 = min(min(e), min(u))
        ymax1 = max(max(e), max(u))
        margin1 = max(0.1, (ymax1 - ymin1) * 0.2)
        # Evitar límites idénticos
        if ymax1 - ymin1 < 0.001:
            margin1 = 0.5
        self.ax1.set_ylim(ymin1 - margin1, ymax1 + margin1)

        # 5. Ajuste de Límites Y dinámico (Subplot 2)
        ymin2 = min(s + [-0.1])
        ymax2 = max(s + [ 0.1])
        margin2 = max(0.1, (ymax2 - ymin2) * 0.2)
        self.ax2.set_ylim(ymin2 - margin2, ymax2 + margin2)

    def show(self):
        plt.tight_layout()
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = PidMonitorPlotter()
    
    # Para mezclar matplotlib (bloqueante) con rclpy (callbacks), 
    # corremos el spin de rclpy en un hilo demonio en background.
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    
    try:
        # plt.show() bloquea el main thread y atiende los eventos de GUI
        node.show()
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt (Ctrl+C). Closing...")
    finally:
        node.get_logger().info("Shutting down monitor plotter...")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
