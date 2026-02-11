#!/usr/bin/env python3
import os
import termios
import tty

import rclpy
from rclpy.node import Node
from qcar2_interfaces.msg import MotorCommands


class TeleopMotorCommands(Node):
    def __init__(self):
        super().__init__('teleop_motorcommands')

        self.pub = self.create_publisher(MotorCommands, 'qcar2_motor_speed_cmd', 10)

        # Estado
        self.speed = 0.0      # m/s deseado (tu driver lo interpreta como desired_speed)
        self.steer = 0.0      # radianes (desired_steering)

        self.speed_step = 0.05
        self.steer_step = 0.05

        # Teclado real
        self.fd = os.open('/dev/tty', os.O_RDONLY)
        self.settings = termios.tcgetattr(self.fd)

        self.get_logger().info(
            "Teleop MotorCommands listo\n"
            "W/S: +/- speed\n"
            "A/D: +/- steer\n"
            "X: stop\n"
            "Q: quit"
        )

        self.timer = self.create_timer(0.02, self.loop)

    def read_key(self):
        try:
            os.set_blocking(self.fd, False)
            tty.setraw(self.fd)
            c = os.read(self.fd, 1)
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.settings)
            if c:
                return c.decode(errors='ignore')
        except BlockingIOError:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.settings)
        return None

    def publish_cmd(self):
        msg = MotorCommands()
        msg.motor_names = ['steering_angle', 'motor_throttle']
        msg.values = [float(self.steer), float(self.speed)]
        self.pub.publish(msg)
        self.get_logger().info(f"qcar2_motor_speed_cmd -> steer={self.steer:.2f} speed={self.speed:.2f}")

    def loop(self):
        k = self.read_key()
        if not k:
            return

        if k == 'w':
            self.speed += self.speed_step
        elif k == 's':
            self.speed -= self.speed_step
        elif k == 'a':
            self.steer += self.steer_step
        elif k == 'd':
            self.steer -= self.steer_step
        elif k == 'x':
            self.speed = 0.0
            self.steer = 0.0
        elif k == 'q':
            raise KeyboardInterrupt
        else:
            return

        # límites razonables
        if self.speed > 0.3: self.speed = 0.3
        if self.speed < -0.3: self.speed = -0.3
        if self.steer > 0.5: self.steer = 0.5
        if self.steer < -0.5: self.steer = -0.5

        self.publish_cmd()

    def destroy_node(self):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.settings)
        os.close(self.fd)
        super().destroy_node()


def main():
    rclpy.init()
    node = TeleopMotorCommands()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
