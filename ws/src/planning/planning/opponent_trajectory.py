"""Accumulate opponent detections into the ProjOppTraj the Gaussian process learns from.

Port of opponent_trajectory.py, which is byte-identical in ForzaETH's Predictive
Spliner and ZJU's FSDP. Both comparison branches import this one copy, so the
GP stages (dense in Predictive Spliner, sparse in FSDP) get the same input, and
any difference between them comes from the GP and not from how each was ported.

Every cycle it takes the nearest dynamic obstacle ahead of the ego, bins it by
the /global_waypoints waypoint it is in, and turns consecutive detections into
ProjOppPoints with finite-difference vs and vd. Once the opponent has travelled
more than half a lap since the last message, the half lap goes out on
/proj_opponent_trajectory and lapcount grows by 0.5.

What it assumes about lap boundaries and staleness, which decides what the GP
sees. All of it is upstream's, kept deliberately:

  * A "lap" is opponent travel, not a start-line crossing. lapcount counts half
    laps of accumulated detection-to-detection distance since the first
    detection, so its boundaries fall wherever the opponent was first seen.
    Detections are at most one per waypoint interval (a new waypoint index is
    required), so a ProjOppTraj is as dense as the waypoints, not the sensor.
  * Before lapcount reaches 1, nothing is ever stale. If the opponent is lost
    and found again further on, the gap is bridged with one point whose vs is
    the average speed across it, and the gap counts towards the half lap. Only
    the very first half lap is checked for gaps: if two consecutive detections
    are more than 2.5 m apart, everything before the gap is dropped, lapcount
    goes back to 0, and the half lap is not published. Later half laps are
    published with whatever gaps they have.
  * From lapcount 1 on, a detection is stale when the node clock is more than
    0.5 s past the header stamp of the last accepted detection. Stale data is
    published (on_trajectory True) if the half lap had more than 5 points, and
    dropped otherwise; either way the half lap restarts.
  * Staleness is the only freshness test, and it reads header stamps. The last
    /perception/obstacles message is reused every cycle whether or not a new
    one came, and is_visible is copied into each point but never checked, so a
    tracker that keeps publishing a lost opponent's predicted position feeds
    those predictions to the GP as detections. The 0.5 s is measured
    from the last detection that moved to a new waypoint, so an opponent that
    is in view but stays inside one waypoint interval for 0.5 s (under about
    0.2 m/s at 0.1 m spacing) goes stale too.
  * The flush adds trav_dist / track_length to lapcount, where trav_dist is
    the distance of the last single step (accepted or not), not the distance
    covered by the half lap. After the first flush lapcount is no longer a
    multiple of 0.5. The GP's `lap_count == 1` test is only reachable before
    any flush, since flushes only happen from 1 on.
  * From lapcount 1 on, a step of more than 1 m between detections is dropped
    and the next detection is taken as a "first point": vs = vd = 0, no
    bounds, speed or direction checks, and the oldest detection of the half
    lap is popped to make room. The gap does not count towards the half lap.
  * The very first detection of a half lap that starts from nothing is
    counted in nrofpoints but not kept (velocity unknown), so nrofpoints can
    exceed len(detections). After a published half lap, its last detection is
    kept as the first detection of the next one, so consecutive messages share
    a point.
  * Off-trajectory: from lapcount 1, once /opponent_trajectory has been
    received twice, each accepted detection is compared with the first
    predicted waypoint with a larger s (first in list order, not nearest). More
    than 5 consecutive detections 0.3 m or more off in d, while under half a
    lap has been covered, publish the half lap so far with on_trajectory
    False, then wait up to 3 s for a new /opponent_trajectory. While waiting,
    detections are still added but nothing is checked for staleness, off-ness
    or lap completion; after it, the off counter resets and the half lap
    continues from where it was, so the points sent as off-trajectory are sent
    again in the next on-trajectory half lap.
  * /opponent_trajectory messages received while the state machine says
    OVERTAKE are not stored, but still count towards the "received twice".
  * The opponent is the dynamic obstacle whose s_start is nearest ahead of the
    ego (modulo the track length) and within 6 m, recorded at its s_center.
    The loop over obstacles resets its pick to None on every obstacle that is
    not a nearer dynamic one, so with several obstacles the opponent is only
    found if the pick is the last in the array. With one obstacle it works as
    intended. Nothing is tracked while the opponent is behind or out of range.
  * Detections going backwards in s by less than half a lap are dropped as
    duplicates. Backwards by more than half a lap is a wrap: vs is copied from
    the previous kept detection, or the point is dropped if the half lap has
    none yet. Exactly half a lap backwards is treated as forwards, with a
    negative vs and distance. Detections off the track (by the waypoint's
    d_left/d_right), faster than 15 m/s, or whose tracker speed changes faster
    than 20 m/s^2 are dropped. Two detections stamped under 10 ms apart are
    dropped and their dt is taken as 0.04 s for the arithmetic.
  * Two track lengths: the one fixed when the loop starts (bins, bounds, the
    half-lap test, the wrap) and the one from the latest /global_waypoints
    (the first-half-lap gap test and its recomputed distance). They only
    differ if the waypoints change while running.
  * When the off-trajectory wait ends on a cycle with no opponent and the half
    lap completes there, the raw row the next detection is differenced against
    is None, which numpy stores as NaN. That makes the next step's distance
    NaN; the half lap's distance is then NaN, never exceeds half a lap, and
    only a stale flush (from lapcount 1) or nothing recovers it.
  * The marker array keeps the last few markers and replaces the oldest two
    with the oldest turned into a DELETEALL. After a few cycles it alternates
    DELETEALL with single markers, so RViz shows only the newest one.

Where upstream would crash it is made not to, as in the other two ports:

  * Upstream raised AttributeError if /opponent_trajectory had been received
    more than once but only ever during OVERTAKE (nothing stored). The
    off-trajectory check is skipped until a trajectory is stored.

No rclpy here, so it runs under plain pytest. opponent_trajectory_node.py is
the wrapper. Times are seconds as floats, on the clock that stamps
/perception/obstacles; the node passes its own clock for "now".
"""

from dataclasses import dataclass, field, replace

import numpy as np

# Upstream's constants, all marked "(variable)" in its source.
STALE_AFTER = 0.5            # s since the last accepted detection, from lapcount 1
MIN_POINTS_TO_FLUSH = 5      # a stale half lap is published only with more points
ON_TRAJECTORY_D = 0.3        # m from the predicted d to count as on it
MAX_POINTS_OFF = 5           # consecutive off points before the half lap is cut
WAIT_FOR_PREDICTION = 3.0    # s to wait for a new /opponent_trajectory after a cut
MAX_STEP = 1.0               # m between detections, from lapcount 1
INITIAL_GAP = 2.5            # m between detections that restarts the first half lap
MAX_SPEED = 15.0             # m/s, projected
MAX_SPEED_CHANGE = 20.0      # m/s^2, of the tracker's speed
MIN_DT = 0.01                # s between stamps; closer is discarded
FALLBACK_DT = 0.04           # s used in its place
SEARCH_AHEAD = 4.0           # m, only matters when no opponent is found
TRACK_AHEAD = 6.0            # m ahead of the ego within which an opponent is taken
MARKER_MAX = 5

# Columns of the raw table, upstream's layout.
S, D, VS, VD, IS_STATIC, IS_VISIBLE, TIME, S_VAR, D_VAR, VS_VAR, VD_VAR = range(11)


@dataclass
class ProjPoint:
    """f1t_msgs/ProjOppPoint fields."""

    s: float = 0.0
    d: float = 0.0
    vs: float = 0.0
    vd: float = 0.0
    is_static: int = 0
    is_visible: int = 0
    time: float = 0.0
    s_var: float = 0.0
    d_var: float = 0.0
    vs_var: float = 0.0
    vd_var: float = 0.0


@dataclass
class ProjTraj:
    """f1t_msgs/ProjOppTraj, as published. detections is a snapshot."""

    lapcount: float
    nrofpoints: int
    opp_is_on_trajectory: bool
    detections: list


@dataclass
class MarkerRec:
    """One entry of the /opponent_marker array. The node places it with to_cartesian."""

    id: int
    s: float
    d: float
    delete_all: bool = False


@dataclass
class Step:
    """What one cycle wants published, in order."""

    trajectories: list = field(default_factory=list)   # ProjTraj
    markers: list = field(default_factory=list)        # snapshots of the marker array


def get_index(s_sorted, opponent_s):
    """First waypoint with s greater than the opponent's, or 0 past the last one."""
    for i, s in enumerate(s_sorted):
        if s > opponent_s:
            return i
    return 0


def find_nearest_dyn_obstacle(obstacles, stamp, ego, track_length):
    """Upstream's opponent pick, including its reset on every non-matching obstacle.

    Returns the 11-element raw row (see the column constants) or None.
    """
    if not (len(obstacles) > 0 and len(ego)):
        return None
    allowed = SEARCH_AHEAD
    closest = track_length
    opponent = None
    for obs in obstacles:
        dist = (obs.s_start - ego[0]) % track_length
        if dist <= closest and not obs.is_static:
            closest = dist
            opponent = [obs.s_center, obs.d_center, obs.vs, obs.vd, obs.is_static,
                        obs.is_visible, stamp, obs.s_var, obs.d_var, obs.vs_var,
                        obs.vd_var]
            allowed = TRACK_AHEAD
        else:
            opponent = None
        if closest > allowed:
            opponent = None
    return opponent


def check_on_trajectory(proj, off_count, oppwpnts):
    """Compare d with the first predicted waypoint ahead in s. Returns the new count."""
    closest = None
    for w in oppwpnts:
        if w.s_m - proj.s > 0 and closest is None:
            closest = w
    if closest is not None:
        if abs(closest.d_m - proj.d) < ON_TRAJECTORY_D:
            off_count = 0
        else:
            off_count += 1
    return off_count


def delta_time(raw, old_index, index):
    dt = raw[index][TIME] - raw[old_index][TIME]
    if dt < MIN_DT:
        return FALLBACK_DT, True
    return dt, False


class OpponentTrajectoryTracker:
    """The state upstream kept on its node and in its loop. step() is one loop pass."""

    def __init__(self):
        # inputs
        self.obstacles = []
        self.obstacle_stamp = 0.0
        self.have_obstacles = False
        self.ego = np.array([])               # s, d, vs, vd; empty until odometry
        self.overtake = False
        self.opponent_trajectory = None       # list of OppWpnt, or None
        self.opp_traj_count = 0
        self.live_track_length = 70.0         # upstream's "feasible" initial value

        # fixed when the loop starts, from the first /global_waypoints
        self.configured = False
        self.ego_s_sorted = []
        self.bound_left = []
        self.bound_right = []
        self.track_length = None
        self.treshold = None
        self.n = 0

        # loop state
        self.lap_count = 0
        self.first_point = True
        self.traversed = 0
        self.index = None
        self.points_in_lap = 0
        self.raw = None
        self.detections = []
        self.initial_lap = True
        self.off_count = 0
        self.trav_dist = 0          # last single step, read by the stale flush
        self.opponent = None        # loop locals that survive into the wait
        self.proj = None
        self.markers = []
        self.marker_count = 0
        self.waiting = None         # (start time, opp_traj_count when it began)

    # ------------------------------------------------------------------ inputs

    @property
    def ready(self):
        return self.configured and self.have_obstacles

    def set_global_waypoints(self, wpnts):
        wpnts = list(wpnts)
        if not wpnts:
            return False    # upstream would raise IndexError here
        self.live_track_length = wpnts[-1].s_m
        if not self.configured:
            self.configured = True
            self.ego_s_sorted = [w.s_m for w in wpnts]
            self.bound_right = [w.d_right for w in wpnts]
            self.bound_left = [w.d_left for w in wpnts]
            self.track_length = self.live_track_length
            self.treshold = self.track_length / 2
            self.n = len(wpnts)
            self.raw = np.zeros((self.n, 11))
        return True

    def set_obstacles(self, obstacles, stamp):
        self.obstacles = list(obstacles)
        self.obstacle_stamp = float(stamp)
        self.have_obstacles = True

    def set_ego(self, s, d, vs, vd):
        self.ego = np.array([s, d, vs, vd])

    def set_state(self, state):
        self.overtake = state == 'OVERTAKE'

    def set_opponent_trajectory(self, oppwpnts):
        if not self.overtake:
            self.opponent_trajectory = list(oppwpnts)
        self.opp_traj_count += 1

    # ------------------------------------------------------------------ loop

    def step(self, now):
        out = Step()
        if self.waiting is not None:
            self._wait_pass(now, out)
            return out

        if self.index is not None and self.lap_count >= 1:
            if now - self.raw[self.index][TIME] > STALE_AFTER:
                if self.points_in_lap > MIN_POINTS_TO_FLUSH:
                    self.lap_count += self.trav_dist / self.track_length
                    out.trajectories.append(self._snapshot(True))
                self._restart(points_in_lap=1)

        self.opponent = self._find()
        if self.opponent is None:
            return out
        old_index = self.index
        self.index = get_index(self.ego_s_sorted, self.opponent[S])
        if self.index == old_index:
            return out

        self._take(old_index, out, check_off=True)

        if self.off_count > MAX_POINTS_OFF and self.traversed < self.track_length / 2:
            out.trajectories.append(self._snapshot(False))
            self.waiting = (now, self.opp_traj_count)
            return out    # the rest of this pass happens when the wait ends

        self._check_half_lap(out)
        return out

    def _wait_pass(self, now, out):
        """One pass of upstream's 3 s wait loop, then its exit if it is due."""
        self.opponent = self._find()
        if self.opponent is not None:
            old_index = self.index
            self.index = get_index(self.ego_s_sorted, self.opponent[S])
            if self.index != old_index:
                self._take(old_index, out, check_off=False)

        start, count = self.waiting
        if self.opp_traj_count != count or not now - start < WAIT_FOR_PREDICTION:
            self.waiting = None
            self.off_count = 0
            self._check_half_lap(out)

    def _find(self):
        return find_nearest_dyn_obstacle(self.obstacles, self.obstacle_stamp, self.ego.copy(),
                                         self.track_length)

    def _take(self, old_index, out, check_off):
        """Record the opponent at self.index and add it to the half lap if it passes."""
        self.raw[self.index] = self.opponent
        self.trav_dist, self.proj, discard, self.first_point = self.project(old_index, self.index)
        if discard:
            self.index = old_index
            return
        if (check_off and self.lap_count >= 1 and self.opp_traj_count > 1
                and self.opponent_trajectory is not None):
            self.off_count = check_on_trajectory(self.proj, self.off_count,
                                                 self.opponent_trajectory)
        self.detections.append(self.proj)
        if self.first_point:
            self.detections.pop(0)
        self.traversed += self.trav_dist
        self.first_point = False
        self.points_in_lap += 1

        mrk = MarkerRec(id=self.marker_count, s=self.proj.s, d=self.proj.d)
        if len(self.markers) > MARKER_MAX:
            oldest = self.markers.pop(0)
            self.markers.pop(0)
            oldest.delete_all = True
            self.markers.append(oldest)
        self.markers.append(mrk)
        out.markers.append([replace(m) for m in self.markers])
        self.marker_count += 1

    def project(self, old_index, index):
        """Upstream's project_opponent_trajectory: (step distance, point, discard, first_point)."""
        raw = self.raw
        first_point = self.first_point
        opp_vs = 0
        opp_vd = 0
        trav_dist = 0
        row = raw[index]
        p = ProjPoint(s=row[S], d=row[D], vs=0, vd=0,
                      is_static=int(row[IS_STATIC]), is_visible=int(row[IS_VISIBLE]),
                      time=row[TIME], s_var=row[S_VAR], d_var=row[D_VAR],
                      vs_var=row[VS_VAR], vd_var=row[VD_VAR])

        discard = False
        if not first_point:
            diff_s = raw[index][S] - raw[old_index][S]
            diff_d = raw[index][D] - raw[old_index][D]
            dt, discard = delta_time(raw, old_index, index)

            if diff_s < 0 and abs(diff_s) < self.treshold:
                discard = True                     # backwards: a duplicate
            elif diff_s < 0 and abs(diff_s) > self.treshold:
                diff_s = diff_s % self.track_length    # across the start line
                if len(self.detections) > 0:
                    opp_vs = self.detections[-1].vs
                else:
                    discard = True
                opp_vd = diff_d / dt
                trav_dist += diff_s
            else:
                opp_vs = diff_s / dt
                opp_vd = diff_d / dt
                trav_dist += diff_s

            p.vs = opp_vs
            p.vd = opp_vd

            if p.d < 0:
                if abs(p.d) > self.bound_right[index]:
                    discard = True
            elif p.d > self.bound_left[index]:
                discard = True

            curr_vel = np.linalg.norm([raw[index][VS], raw[index][VD]])
            old_vel = np.linalg.norm([raw[old_index][VS], raw[old_index][VD]])
            change = abs(curr_vel - old_vel) / dt
            if not discard and (np.sqrt(opp_vs ** 2 + opp_vd ** 2) > MAX_SPEED
                                or change > MAX_SPEED_CHANGE):
                discard = True

            if diff_s > MAX_STEP and self.lap_count >= 1:
                discard = True
                first_point = True

        return trav_dist, p, discard, first_point

    def _check_half_lap(self, out):
        """Publish once more than half a lap is covered; trim the first half lap at a gap."""
        if not self.traversed > self.track_length / 2:
            return
        self.lap_count += 0.5

        if self.initial_lap:
            self.initial_lap = False
            dets = self.detections
            for i in range(len(dets) - 1):
                if (dets[i + 1].s - dets[i].s) % self.live_track_length > INITIAL_GAP:
                    self.initial_lap = True
                    start = i + 1
                    break
            if self.initial_lap:
                del dets[:start]
                if dets:
                    self.index = get_index(self.ego_s_sorted, dets[-1].s)
                    self.points_in_lap -= start
                    self.lap_count = 0
                    self.traversed = (dets[-1].s - dets[0].s) % self.live_track_length
                else:
                    # Unreachable: start <= len - 1 leaves a detection. Kept as upstream.
                    self._restart(points_in_lap=0)
                    self.lap_count = 0

        if not self.initial_lap:
            out.trajectories.append(self._snapshot(True))
            self.detections = [self.proj]
            self.raw = np.zeros((self.n, 11))
            # Upstream assigns the opponent even when it is None, which numpy stores as NaN.
            self.raw[self.index] = self.opponent if self.opponent is not None else np.nan
            self.points_in_lap = 1
            self.traversed = 0

    def _restart(self, points_in_lap):
        self.detections = []
        self.raw = np.zeros((self.n, 11))
        self.traversed = 0
        self.points_in_lap = points_in_lap
        self.first_point = True
        self.index = None

    def _snapshot(self, on_trajectory):
        return ProjTraj(lapcount=self.lap_count, nrofpoints=self.points_in_lap,
                        opp_is_on_trajectory=on_trajectory, detections=list(self.detections))
