#!/usr/bin/env python3
"""Keyboard teleop: publishes /drive (AckermannDriveStamped) at a steady rate.

Speed is in m/s and goes through the adapter's speed loop (the VESC on the car), steering in rad.
Ported from AutoDRIVE's devkit teleop, which wrote raw throttle/steering commands instead.

  ros2 run autodrive_bridge teleop_keyboard    (needs a terminal; keys are read from stdin)
"""
import os
import select
import signal
import sys
import termios
import time
import tty

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.signals import SignalHandlerOptions
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool

HELP = """
---------------------------------------
RoboRacer keyboard teleop -> /drive
---------------------------------------
        Q   W   E
        A   S   D
            X       R

W/S : speed +/- {speed_step:.2f} m/s   (max {max_speed:.1f})
A/D : steer left/right {steer_step:.2f} rad (max {max_steer:.3f})
Q   : zero steering
E   : brake (speed 0)
X   : stop and zero steering
R   : stop and soft-reset the simulator
Ctrl+C : quit (sends a zero command)
---------------------------------------
"""


def clamp(x, limit):
    return max(-limit, min(limit, x))


def main():
    if not sys.stdin.isatty():
        print('teleop_keyboard needs an interactive terminal (use ros2 run, not ros2 launch)')
        return

    # No rclpy SIGINT handler: Ctrl+C raises KeyboardInterrupt and the context stays valid
    # long enough to send the zero command.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGHUP, lambda *_: sys.exit(0))

    node = rclpy.create_node('teleop_keyboard')
    p = node.declare_parameter
    rate = p('rate', 20.0).value                      # Hz; adapter stops after cmd_timeout = 0.5 s
    max_speed = p('max_speed', 5.0).value             # m/s
    speed_step = p('speed_step', 0.5).value
    vehicle = os.path.join(get_package_share_directory('control'), 'config', 'vehicle.yaml')
    with open(vehicle) as f:
        lock = yaml.safe_load(f)['/**']['ros__parameters']['max_steering_angle']
    max_steer = p('max_steering_angle', lock).value  # rad, default from control/config/vehicle.yaml
    steer_step = p('steer_step', 0.1).value
    drive_pub = node.create_publisher(AckermannDriveStamped, '/drive', 10)
    reset_pub = node.create_publisher(Bool, '/autodrive/reset_command', 1)

    def publish(speed, steer):
        msg = AckermannDriveStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steer)
        drive_pub.publish(msg)

    print(HELP.format(speed_step=speed_step, max_speed=max_speed,
                      steer_step=steer_step, max_steer=max_steer))
    fd = sys.stdin.fileno()
    settings = termios.tcgetattr(fd)
    speed = steer = 0.0
    shown = None
    period = 1.0 / rate
    next_t = time.monotonic()
    try:
        tty.setcbreak(fd)
        while True:
            ready, _, _ = select.select([fd], [], [], max(0.0, next_t - time.monotonic()))
            # os.read, not sys.stdin.read: the buffered reader would hold back keys select can't see
            for key in (os.read(fd, 32).decode(errors='ignore').lower() if ready else ''):
                if key == 'w':
                    speed = clamp(speed + speed_step, max_speed)
                elif key == 's':
                    speed = clamp(speed - speed_step, max_speed)
                elif key == 'a':
                    steer = clamp(steer + steer_step, max_steer)
                elif key == 'd':
                    steer = clamp(steer - steer_step, max_steer)
                elif key == 'q':
                    steer = 0.0
                elif key == 'e':
                    speed = 0.0
                elif key in ('x', 'r'):
                    speed = steer = 0.0
                    if key == 'r':
                        reset_pub.publish(Bool(data=True))
            if (speed, steer) != shown:
                shown = (speed, steer)
                print(f'\rspeed {speed:+5.2f} m/s   steer {steer:+6.3f} rad   ', end='', flush=True)
            if time.monotonic() >= next_t:
                publish(speed, steer)
                next_t += period
                if next_t < time.monotonic():   # fell behind, don't burst
                    next_t = time.monotonic() + period
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, settings)
        for _ in range(5):
            publish(0.0, 0.0)
            time.sleep(0.02)
        print('\nstopped, sent zero command')
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
