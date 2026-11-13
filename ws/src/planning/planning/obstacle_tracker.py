"""Multi-obstacle tracker in Frenet coordinates: identity, velocity and variance
for detections that carry only a position.

A detector reports where things are this cycle. The overtaking stack needs more:
opponent_trajectory and collision_prediction ignore anything whose is_static is
True, the GP wants a velocity and divides by the variances, and a planner that
commits to a side wants the same obstacle to keep the same id. None of that
can come from a single scan. This module supplies it from the sequence of them.

Nothing here imports rclpy; obstacle_tracker_node.py is the ROS wrapper.

WHAT IS READ FROM A DETECTION. s_center, d_center, the footprint (s_start,
s_end, d_right, d_left, size), is_actually_a_gap and is_visible. Nothing else:
id, vs, vd, is_static and all four *_var fields are ignored, because a
scan-based detector has no source for them and fills them with constants (the
fake publisher fills them with the answer). Measurement noise is a parameter
(sigma_s, sigma_d). A detection with is_visible False is a memory, not a
measurement, and is skipped; the tracker coasts its own tracks instead.

PER CYCLE (step):
  1. Time. dt is the difference of header stamps. Time going backwards (a
     reset, a bag looping) or a gap over reset_gap s drops every track.
  2. Predict every track with a constant-velocity Kalman filter, state
     (s, d, vs, vd), white-noise acceleration of standard deviation accel_sigma.
  3. Associate, greedy nearest neighbour in (s, d): all track/detection pairs
     inside the gate, nearest first, each used once, confirmed tracks before
     tentative ones. Without that cascade a duplicate is born this way: the
     car's detection drops while a spurious return falls in its gate and is
     absorbed; the car's next detection lands outside the moved gate and starts
     a tentative track; on the next cycle both gate the car, the tentative one
     (sitting on the last detection) is nearer, takes it, and is confirmed.
     Distance is to the
     track's prediction, with s differenced by FrenetConverter.ds so the start
     line is not a 1-lap jump. The gate is the furthest the obstacle could have
     moved since the track was last updated: max_speed * elapsed + gate_margin,
     measured from that update, so it widens while a track coasts.
  4. Update matched tracks. An unmatched tentative track is dropped; an
     unmatched confirmed one coasts, and is dropped after more than max_misses
     consecutive cycles. An unmatched detection starts a tentative track, which
     is confirmed, and given an id, on its confirm_frames-th consecutive match.
  5. Classify. is_static compares the norm of the mean estimated velocity over
     the last static_window s with two thresholds (hysteresis). Until a track
     has existed for static_window s it is not static: hiding a moving car from
     the opponent half of the stack is the costlier mistake.
  6. Output the confirmed tracks in the order described under ORDERING.

LIFECYCLE DEFAULTS, at the 40 Hz fake_opponent_node and lidar rate:
  confirm_frames = 3 (75 ms). One spurious return, or one that recurs for two
    scans, never becomes an obstacle. Three consecutive frames is also the first
    point at which the filter's velocity is an average rather than one finite
    difference. At a 10 m/s closing speed the delay costs 0.75 m of a ~10 m
    detection range.
  max_misses = 10 (0.25 s). Survives the dropouts the sensor model produces
    (at detection_prob 0.8, eleven misses in a row happen with probability
    2e-8 per cycle) and a car briefly hidden behind another. Coasting longer
    costs accuracy (half of 3 m/s^2 over 1 s is 1.5 m) and, because
    opponent_trajectory counts invisible obstacles as detections (see
    docs/forzaeth-quirks.md), feeds the GP predictions instead of sightings.
  Both are counted in cycles, so they scale with the input rate: at the 15 Hz
  of a scan-rate spliner they are 0.2 s and 0.67 s.

ORDERING. collision_prediction takes the FIRST obstacle that is not static as
the opponent; opponent_trajectory takes the LAST obstacle, and only if it is
dynamic and no nearer-ahead dynamic obstacle precedes it (its distance is
(s_start - ego_s) % lap). With two or more dynamic obstacles no ordering makes
them pick the same one: the first needs the target before the other dynamic
obstacles, the second after them. The output is: static obstacles first (by
s), then dynamic ones by (s_start - ego_s) % lap ascending, i.e. nearest ahead
first. With one dynamic obstacle both consumers take it. With more,
collision_prediction takes the nearest one ahead and opponent_trajectory takes
none, so they never follow two different cars. Statics must come first:
opponent_trajectory resets its pick on any obstacle after it. Without an ego
position the dynamic ones are ordered by id. See docs/forzaeth-quirks.md.
"""

from dataclasses import dataclass, field

import numpy as np

from planning.opponent_models import VAR_FLOOR, ObstacleObservation


@dataclass
class TrackerParams:
    max_speed: float = 15.0        # m/s an obstacle can move: the association gate
    gate_margin: float = 0.3       # m added to the gate: detection noise, footprint jitter
    confirm_frames: int = 3        # N: consecutive matched cycles before a track is published
    max_misses: int = 10           # M: consecutive unmatched cycles a confirmed track survives
    sigma_s: float = 0.05          # m, detection noise along the reference line
    sigma_d: float = 0.05          # m, across it
    accel_sigma: float = 2.0       # m/s^2, the filter's process noise
    init_speed_sigma: float = 5.0  # m/s, prior on the velocity of a new track
    static_window: float = 0.5     # s of estimated velocity averaged for is_static
    static_speed: float = 0.3      # m/s: below this the track becomes static
    moving_speed: float = 0.6      # m/s: above this it becomes dynamic again
    reset_gap: float = 1.0         # s without input after which every track is dropped


@dataclass
class Track:
    x: np.ndarray                  # s (wrapped), d, vs, vd
    P: np.ndarray                  # 4 x 4 covariance
    born: float                    # stamp of the first detection
    updated: float                 # stamp of the last matched detection
    s_update: float                # s, d of the track at that update: the gate's centre
    d_update: float
    half_l: float                  # footprint about the centre, from the last detection
    half_w: float
    size: float
    is_gap: bool
    hits: int = 1                  # consecutive matches while tentative
    misses: int = 0                # consecutive misses while confirmed
    id: int = -1                   # -1 while tentative
    visible: bool = True
    is_static: bool = False
    history: list = field(default_factory=list)   # (stamp, vs, vd) of matched cycles

    @property
    def confirmed(self):
        return self.id >= 0


def _finite(*values):
    return all(np.isfinite(v) for v in values)


class ObstacleTracker:
    """Feed it one detection array per cycle with step(); it returns the tracks."""

    def __init__(self, conv, params=None):
        self.conv = conv                      # FrenetConverter: ds and the lap length
        self.lap = float(conv.length)
        self.p = params or TrackerParams()
        self.R = np.diag([self.p.sigma_s ** 2, self.p.sigma_d ** 2])
        self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        self.reset()

    def reset(self):
        self.tracks = []
        self.stamp = None
        self.next_id = 0

    # ------------------------------------------------------------------ filter

    def _predict(self, t, dt):
        if dt <= 0.0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = self.p.accel_sigma ** 2
        a, b, c = dt ** 4 / 4, dt ** 3 / 2, dt ** 2
        Q = q * np.array([[a, 0, b, 0], [0, a, 0, b], [b, 0, c, 0], [0, b, 0, c]])
        t.x = F @ t.x
        t.x[0] %= self.lap
        t.P = F @ t.P @ F.T + Q

    def _update(self, t, s, d):
        y = np.array([float(self.conv.ds(t.x[0], s)), d - t.x[1]])   # circular in s
        S = self.H @ t.P @ self.H.T + self.R
        K = t.P @ self.H.T @ np.linalg.inv(S)
        t.x = t.x + K @ y
        t.x[0] %= self.lap
        IKH = np.eye(4) - K @ self.H
        t.P = IKH @ t.P @ IKH.T + K @ self.R @ K.T   # Joseph form: stays symmetric positive

    def _new_track(self, det, stamp):
        v = self.p.init_speed_sigma ** 2
        t = Track(x=np.array([det.s_center % self.lap, det.d_center, 0.0, 0.0]),
                  P=np.diag([self.p.sigma_s ** 2, self.p.sigma_d ** 2, v, v]),
                  born=stamp, updated=stamp, s_update=det.s_center % self.lap,
                  d_update=det.d_center, half_l=0.0, half_w=0.0, size=0.0, is_gap=False)
        self._take_footprint(t, det)
        return t

    def _take_footprint(self, t, det):
        half_l = (det.s_end - det.s_start) / 2.0
        half_w = (det.d_left - det.d_right) / 2.0
        if _finite(half_l, half_w) and half_l >= 0 and half_w >= 0:
            t.half_l, t.half_w = float(half_l), float(half_w)
        if _finite(det.size):
            t.size = float(det.size)
        t.is_gap = bool(det.is_actually_a_gap)

    # ------------------------------------------------------------------ cycle

    def step(self, detections, stamp, ego_s=None):
        """Advance to stamp (s) with this cycle's detections; return the published
        obstacles (ObstacleObservation, field for field f1t_msgs/Obstacle) in
        output order. ego_s orders the dynamic ones; None orders them by id."""
        stamp = float(stamp)
        if self.stamp is not None and (stamp < self.stamp or stamp - self.stamp > self.p.reset_gap):
            self.reset()
        dt = 0.0 if self.stamp is None else stamp - self.stamp
        self.stamp = stamp

        dets = [o for o in detections
                if getattr(o, 'is_visible', True) and _finite(o.s_center, o.d_center)]

        for t in self.tracks:
            self._predict(t, dt)

        pairs = []
        for i, t in enumerate(self.tracks):
            reach = self.p.max_speed * (stamp - t.updated) + self.p.gate_margin
            for j, o in enumerate(dets):
                moved = np.hypot(float(self.conv.ds(t.s_update, o.s_center)), o.d_center - t.d_update)
                if moved > reach:
                    continue
                dist = np.hypot(float(self.conv.ds(t.x[0], o.s_center)), o.d_center - t.x[1])
                pairs.append((not t.confirmed, dist, i, j))
        pairs.sort()   # confirmed tracks first: a new hypothesis never takes an established one's detection
        track_of, det_used = {}, set()
        for _, _, i, j in pairs:
            if i not in track_of and j not in det_used:
                track_of[i] = j
                det_used.add(j)

        survivors = []
        for i, t in enumerate(self.tracks):
            if i in track_of:
                o = dets[track_of[i]]
                self._update(t, o.s_center, o.d_center)
                self._take_footprint(t, o)
                t.updated, t.s_update, t.d_update = stamp, o.s_center % self.lap, o.d_center
                t.visible, t.misses = True, 0
                if not t.confirmed:
                    t.hits += 1
                    if t.hits >= self.p.confirm_frames:
                        t.id, self.next_id = self.next_id, self.next_id + 1
                t.history.append((stamp, float(t.x[2]), float(t.x[3])))
                self._classify(t, stamp)
                survivors.append(t)
            elif t.confirmed:
                t.visible = False
                t.misses += 1
                if t.misses <= self.p.max_misses:
                    survivors.append(t)
            # an unmatched tentative track is dropped: confirmation needs consecutive frames

        for j, o in enumerate(dets):
            if j not in det_used:
                t = self._new_track(o, stamp)
                if self.p.confirm_frames <= 1:
                    t.id, self.next_id = self.next_id, self.next_id + 1
                survivors.append(t)   # no velocity yet: its (0, 0) is a prior, not an estimate
        self.tracks = survivors
        return self._output(ego_s)

    def _classify(self, t, stamp):
        t.history = [h for h in t.history if stamp - h[0] <= self.p.static_window]
        if stamp - t.born < self.p.static_window:
            t.is_static = False
            return
        speed = float(np.hypot(np.mean([h[1] for h in t.history]),
                               np.mean([h[2] for h in t.history])))
        if t.is_static and speed > self.p.moving_speed:
            t.is_static = False
        elif not t.is_static and speed < self.p.static_speed:
            t.is_static = True

    # ------------------------------------------------------------------ output

    def _obstacle(self, t):
        s, d = float(t.x[0]) % self.lap, float(t.x[1])
        s_start, s_end = s - t.half_l, s + t.half_l
        if s_start < 0.0:   # s_start < s_end across the start line: s_end runs past the lap
            s_start += self.lap
            s_end += self.lap
        var = np.maximum(np.diag(t.P), VAR_FLOOR)
        return ObstacleObservation(
            id=int(t.id), s_start=s_start, s_end=s_end,
            d_right=d - t.half_w, d_left=d + t.half_w, is_actually_a_gap=t.is_gap,
            s_center=s, d_center=d, size=t.size, vs=float(t.x[2]), vd=float(t.x[3]),
            s_var=float(var[0]), d_var=float(var[1]), vs_var=float(var[2]), vd_var=float(var[3]),
            is_static=bool(t.is_static), is_visible=bool(t.visible))

    def _output(self, ego_s):
        out = [self._obstacle(t) for t in self.tracks if t.confirmed]
        static = sorted((o for o in out if o.is_static), key=lambda o: (o.s_center, o.id))
        moving = [o for o in out if not o.is_static]
        if ego_s is None:
            moving.sort(key=lambda o: o.id)
        else:
            moving.sort(key=lambda o: ((o.s_start - ego_s) % self.lap, o.id))
        return static + moving
