"""Overwrite the scaled raceline's speed profile with the speed the car actually drives.

Port of update_waypoints.py, which is byte-identical in ForzaETH's Predictive
Spliner and ZJU's FSDP. Both comparison branches import this one copy, so any
difference between them comes from the planners and not from how each was ported.

While the state machine has been in GB_TRACK (following the global raceline) for
at least hysteresis_time, every odometry message overwrites vx_mps of the waypoint
nearest the car with the car's measured speed plus speed_offset. The collision
predictor then reads these speeds instead of the planned ones, so it predicts
when the car will reach a point from what the car really does.

The behaviour is upstream's, including its rough edges, because changing it here
would make both branches differ from the code they claim to port:

  * The nearest waypoint is found by |s_m - ego_s|, with no lap wrap. The
    raceline's first and last waypoints are the same point, so near the line
    either one can be picked.
  * Upstream meant to keep those two endpoints equal when either is written,
    but the two lines that would do it only read the values and assign nothing.
    That is preserved: writing one endpoint leaves the other alone.
  * Only the first scaled and global waypoint arrays are used; upstream read
    each with wait_for_message once at startup. A one-shot publish that
    happens before this node subscribes is therefore missed; see
    update_waypoints_node.py on QoS durability.
  * Starting state is GB_TRACK with no start time, so nothing is updated until
    a GB_TRACK message actually arrives.

Upstream also opened a proxy to convert_frenet2globarr_service and never called
it. Nothing replaces it here because nothing used it.

No rclpy here, so it runs under plain pytest. update_waypoints_node.py is the wrapper.
"""

import numpy as np

GB_TRACK = 'GB_TRACK'


class WaypointUpdater:
    """The state upstream kept on its node, with time passed in rather than read."""

    def __init__(self, hysteresis_time=2.0, speed_offset=0.0, enabled=True):
        self.hysteresis_time = float(hysteresis_time)
        self.speed_offset = float(speed_offset)
        self.enabled = bool(enabled)    # upstream's update_waypoints flag

        self.state = GB_TRACK
        self.gb_track_start_time = None

        self.wpnts_msg = None           # the scaled WpntArray, edited in place
        self.s_points = np.array([])
        self.max_speed = 10.0           # m/s, only scales the markers
        self.have_global = False

    @property
    def ready(self):
        return self.have_global and self.wpnts_msg is not None and self.s_points.size > 0

    def set_scaled(self, msg):
        """Take the first scaled waypoint array. Later ones are ignored, as upstream."""
        if self.wpnts_msg is not None:
            return False
        self.wpnts_msg = msg
        self.s_points = np.array([w.s_m for w in msg.wpnts])
        return True

    def set_global(self, msg):
        """Take the top speed of the first unscaled raceline, used to scale the markers."""
        if self.have_global or not msg.wpnts:
            return False
        self.max_speed = max(w.vx_mps for w in msg.wpnts)
        self.have_global = True
        return True

    def on_state(self, state, now):
        self.state = state
        if state == GB_TRACK and self.gb_track_start_time is None:
            self.gb_track_start_time = now
        elif state != GB_TRACK:
            self.gb_track_start_time = None

    def tracking_long_enough(self, now):
        return (self.state == GB_TRACK and self.gb_track_start_time is not None
                and now - self.gb_track_start_time >= self.hysteresis_time)

    def on_odom(self, ego_s, ego_speed, now):
        """Write the measured speed into the nearest waypoint.

        Returns the index written, or None when nothing was. ego_s and ego_speed
        are s and vs from /car_state/odom_frenet.
        """
        if not self.enabled or not self.ready:
            return None
        idx = int(np.abs(self.s_points - ego_s).argmin())
        if not self.tracking_long_enough(now):
            return None
        self.wpnts_msg.wpnts[idx].vx_mps = float(ego_speed) + self.speed_offset
        return idx

    def marker_heights(self):
        """Cylinder height per waypoint, vx_mps over the raceline's top speed."""
        if not self.ready:
            return []
        return [w.vx_mps / self.max_speed for w in self.wpnts_msg.wpnts]
