"""The QoS of /global_waypoints and /global_waypoints_scaled, shared by both ends.

The waypoints are published once, latched: TRANSIENT_LOCAL, KEEP_LAST(1), RELIABLE.
A node that starts after the publisher still gets them, but only if its
subscription is TRANSIENT_LOCAL too. A VOLATILE subscriber is compatible with a
TRANSIENT_LOCAL publisher and silently receives nothing published before it
joined, which is exactly how the shared ports used to hang. So the publisher
and every subscriber import this one profile.

The flip side: a TRANSIENT_LOCAL subscriber is incompatible with a VOLATILE
publisher. To publish waypoints by hand, match it:

    ros2 topic pub --once --qos-durability transient_local --qos-depth 1 \\
        /global_waypoints f1t_msgs/msg/WpntArray "{...}"
"""

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

WAYPOINTS_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)
