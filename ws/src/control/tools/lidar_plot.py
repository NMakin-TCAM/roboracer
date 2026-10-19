import numpy as np
import matplotlib
matplotlib.use("WebAgg")
matplotlib.rcParams["webagg.open_in_browser"] = False
import matplotlib.pyplot as plt
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from control.lidar_processing import LidarProcessor

rclpy.init()
node = rclpy.create_node("lidar_plot")
# Use the same parameter names/defaults as the controller; override with --ros-args -p.
processor = LidarProcessor(**{
    name: node.declare_parameter(name, default).value
    for name, default in {
        "max_range": 5.0,
        "disparity_threshold": 2.0,
        "alpha": 0.7,
        "car_width": 0.27,
    }.items()
}) # type: ignore

plt.ion()
fig, ax = plt.subplots()
raw_line, = ax.plot([], [], label="Raw", alpha=0.5)
processed_line, = ax.plot([], [], label="Processed")
steering_line = ax.axvline(
    0, color="red", linestyle=":", label="Steering"
)
ax.legend()
ax.set(xlabel="θ (degrees)", ylabel="Range (m)", title="/scan")
ax.invert_xaxis() # since negative angle is to the right
ax.set_ylim(0, 5)
ax.grid(True)

def update(msg):
    processed = processor.process(msg)
    ranges = np.asarray(msg.ranges, dtype=float)
    theta = np.degrees(
        msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
    )
    invalid = (
        ~np.isfinite(ranges)
        | (ranges < msg.range_min)
        | (ranges > msg.range_max)
    )
    ranges[invalid] = np.nan
    raw_line.set_data(theta, ranges)
    processed_line.set_data(theta, processed)
    ax.relim()
    ax.autoscale_view(scaley=False)
    fig.canvas.draw_idle()

def update_steering(msg: AckermannDriveStamped):
    steering_deg = np.degrees(msg.drive.steering_angle)
    # plot vertical line
    steering_line.set_xdata([steering_deg, steering_deg])

sub = node.create_subscription(
    LaserScan, "/scan", update, qos_profile_sensor_data
)
drive_sub = node.create_subscription(
    AckermannDriveStamped, "/drive", update_steering, qos_profile_sensor_data
)

timer = fig.canvas.new_timer(interval=30)
timer.add_callback(lambda: rclpy.spin_once(node, timeout_sec=0))
timer.start()

try:
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    timer.stop()
    node.destroy_node()
    rclpy.shutdown()
