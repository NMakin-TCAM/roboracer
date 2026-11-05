"""Lap timing and per-lap statistics from the car's pose along the raceline (no ROS; lap_logger_node uses it).

A lap is timed from one crossing of the raceline's start (s = 0) to the next, interpolated between the two
poses either side of the line. It only counts if the car passed the halfway point (s in [0.4, 0.6] of the
lap) since the last crossing, so pose noise at the line or reversing over it never counts a lap.

The first lap of a run is a standing start: if the car begins just after the line (s < 25 % of the lap),
it is timed from when the car first moves; otherwise from the first crossing. Later laps are flying.

A lap is aborted, not finished, when the pose jumps more than max_jump between two poses (a sim reset,
a respawn, a relocalization) or the node calls abort() (the sim reported a reset). The tracker then
starts over as at the beginning of a run.

Statistics per lap: time, distance driven, mean and max distance of the car from the raceline (as the
localizer sees it: pure pursuit's tracking error, not the true one), and max speed between poses.
"""
import math
from dataclasses import dataclass, field

import numpy as np

START_WINDOW = 0.25    # fraction of the lap after the line where a run may begin as a standing start
HALF = (0.4, 0.6)      # fraction of the lap that counts as the halfway point
MOVED = 0.1            # m from the first pose: the car has started moving


@dataclass
class Lap:
    number: int                 # 1, 2, ... in the run, aborted laps included
    status: str                 # 'ok' or 'aborted'
    start: str                  # 'standing' or 'flying'
    time_s: float               # lap time; for an aborted lap, the time until the abort
    distance_m: float
    mean_offset_m: float
    max_offset_m: float
    max_speed_mps: float
    note: str = ''              # why it was aborted

    @property
    def mean_speed_mps(self):
        return self.distance_m / self.time_s if self.time_s > 0 else 0.0


@dataclass
class _Acc:
    t0: float
    start: str
    distance: float = 0.0
    offsets: list = field(default_factory=list)
    max_speed: float = 0.0


class LapTracker:
    def __init__(self, xy, max_jump=1.0):
        """xy: raceline points (N x 2, closed loop, the last point not repeating the first)."""
        xy = np.asarray(xy, dtype=float)
        if len(xy) > 1 and np.allclose(xy[0], xy[-1]):
            xy = xy[:-1]
        if len(xy) < 3:
            raise ValueError('a raceline needs at least 3 points')
        self.a = xy
        self.d = np.roll(xy, -1, axis=0) - xy
        self.seg_len = np.hypot(self.d[:, 0], self.d[:, 1])
        self.seg_len2 = np.maximum(self.seg_len ** 2, 1e-12)
        self.s0 = np.concatenate(([0.0], np.cumsum(self.seg_len)[:-1]))
        self.length = float(self.seg_len.sum())
        self.max_jump = max_jump
        self.laps = []           # every Lap of the run, in order
        self._reset()

    def _reset(self):
        self.first = None        # (x, y) of the first pose since the start or the last abort
        self.prev = None         # (t, x, y, s) of the last pose
        self.acc = None          # the lap being timed, or None
        self.armed = False       # standing start: time from the first movement
        self.half = False

    def project(self, x, y):
        """(s, offset): arc length of the closest raceline point and the car's distance from it."""
        p = np.array([x, y])
        t = np.clip(((p - self.a) * self.d).sum(axis=1) / self.seg_len2, 0.0, 1.0)
        q = self.a + t[:, None] * self.d
        dist = np.hypot(q[:, 0] - x, q[:, 1] - y)
        i = int(np.argmin(dist))
        return float(self.s0[i] + t[i] * self.seg_len[i]), float(dist[i])

    def update(self, t, x, y):
        """Feed one pose (time in s, map-frame position). Returns the Lap it finished or aborted, else None."""
        s, offset = self.project(x, y)
        L = self.length
        if self.prev is None:
            self.first = (x, y)
            self.prev = (t, x, y, s)
            self.armed = s < START_WINDOW * L
            return None
        t1, x1, y1, s1 = self.prev
        self.prev = (t, x, y, s)
        step = math.hypot(x - x1, y - y1)
        if step > self.max_jump:
            return self.abort(f'pose jumped {step:.1f} m')

        if self.armed and math.hypot(x - self.first[0], y - self.first[1]) > MOVED:
            self.armed = False
            self.acc = _Acc(t1, 'standing')   # from the last pose before the car moved

        finished = None
        crossed = s1 > (1 - START_WINDOW) * L and s < START_WINDOW * L
        if crossed:
            ahead = L - s1 + s
            t_cross = t1 + (t - t1) * ((L - s1) / ahead if ahead > 0 else 1.0)
            if self.acc is not None and self.half:
                finished = self._close(t_cross, 'ok')
            if self.acc is None or finished is not None:
                # first crossing of the run (standing if the car was still slow there), or the next
                # lap starts where this one ended
                slow = finished is None and step / max(t - t1, 1e-6) < 0.5
                self.acc = _Acc(t_cross, 'standing' if slow else 'flying')
                self.armed = False
            self.half = False
        elif HALF[0] * L <= s <= HALF[1] * L:
            self.half = True

        if self.acc is not None:
            self.acc.distance += step
            self.acc.offsets.append(offset)
            if t > t1:
                self.acc.max_speed = max(self.acc.max_speed, step / (t - t1))
        return finished

    def abort(self, reason):
        """End the lap being timed as aborted (returned, None if none) and start over."""
        lap = None
        if self.acc is not None:
            lap = self._close(self.prev[0], 'aborted', reason)
        self._reset()
        return lap

    def _close(self, t, status, note=''):
        a = self.acc
        offs = a.offsets or [0.0]
        lap = Lap(len(self.laps) + 1, status, a.start, t - a.t0, a.distance,
                  float(np.mean(offs)), float(np.max(offs)), a.max_speed, note)
        self.laps.append(lap)
        self.acc = None
        return lap


def summary(laps):
    """One line: laps finished and aborted, best and mean time of the finished ones."""
    ok = [lap.time_s for lap in laps if lap.status == 'ok']
    flying = [lap.time_s for lap in laps if lap.status == 'ok' and lap.start == 'flying']
    aborted = sum(lap.status == 'aborted' for lap in laps)
    if not ok:
        return f'no finished laps, {aborted} aborted'
    text = f'{len(ok)} laps, best {min(ok):.2f} s'
    if flying:
        text += f', flying mean {sum(flying) / len(flying):.2f} s'
    return text + f', {aborted} aborted'
