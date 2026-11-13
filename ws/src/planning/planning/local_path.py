"""Turn the spliner's evasion into the /local_path pure pursuit follows. Imports no ROS.

The seam. control's pure_pursuit_node with avoid:=true follows /local_path
(nav_msgs/Path, map frame) and /local_path/valid (std_msgs/Bool). It accepts a
path only with exactly one pose per raceline row, the TUM closing row dropped,
and it stops on a missing, stale (max_path_age) or invalid one. It drives the
raceline's speeds whatever the planner wants. obstacle_spliner_node publishes
that seam; spliner_node publishes /planner/avoidance/otwpnts (OTWpntArray: only
the manoeuvre, empty when not evading) and /planner/avoidance/status. This
module is the adapter between them, and local_path_node.py its ROS wrapper.

SPLICE. Every raceline row keeps its exact x, y, except the rows whose s falls
inside the evasion. Those move to the evasion's d at that s, interpolated in
arc length from the evasion's first point, so the wrap at the start line is
handled once.

VALIDITY. The path is valid only when all of these hold:

    a status paired with its trajectory (same stamp) has arrived
    that pair arrived within max_plan_age
    the ego state arrived within max_ego_age
    the status is one plan() knows, and is collision-free (CLEAR or EVADING)
    the trajectory agrees with the status: empty for CLEAR, non-empty for EVADING

Anything else is invalid and pure pursuit stops. NO_ROOM and TOO_CLOSE are the
cases that matter most: their trajectory is empty, exactly like CLEAR's, and an
adapter that read an empty trajectory as "follow the raceline" would drive the
car into the obstacle the planner refused.
"""

from dataclasses import dataclass

import numpy as np

from planning.raceline import read_raceline
from planning.spliner import CLEAR, COLLISION_FREE, EVADING, STATUSES

# raceline.csv columns: s_m; x_m; y_m; ...
_S, _X, _Y = 0, 1, 2


def load_raceline_rows(path):
    """(s, xy) of the raceline rows pure pursuit indexes: the closing row dropped, as it does."""
    rows = read_raceline(path)
    if len(rows) > 1 and np.allclose(rows[0, [_X, _Y]], rows[-1, [_X, _Y]]):
        rows = rows[:-1]
    return rows[:, _S].copy(), rows[:, [_X, _Y]].copy()


def splice(raceline_s, raceline_xy, lap, evasion_s, evasion_d, to_cartesian):
    """(N, 2) path: the raceline with the evasion's stretch replaced. Empty evasion -> raceline.

    evasion_s runs forward from the car and may wrap past lap; evasion_d is metres left of
    the raceline. to_cartesian(s, d) -> (x, y) is the raceline's FrenetConverter.
    """
    xy = np.array(raceline_xy, dtype=float, copy=True)
    evasion_s = np.asarray(evasion_s, dtype=float)
    evasion_d = np.asarray(evasion_d, dtype=float)
    if len(evasion_s) == 0:
        return xy
    if len(evasion_s) != len(evasion_d):
        raise ValueError('evasion s and d differ in length')
    u_ev = np.concatenate(([0.0], np.cumsum(np.diff(evasion_s) % lap)))
    u = (np.asarray(raceline_s, dtype=float) - evasion_s[0]) % lap
    inside = u <= u_ev[-1]
    if np.any(inside):
        d = np.interp(u[inside], u_ev, evasion_d)
        x, y = to_cartesian(np.asarray(raceline_s)[inside], d)
        xy[inside, 0], xy[inside, 1] = x, y
    return xy


@dataclass
class Plan:
    """A trajectory and its status, paired by stamp. received: node-clock seconds."""

    status: str
    note: str
    s: np.ndarray
    d: np.ndarray
    received: float


@dataclass
class Decision:
    valid: bool
    reason: str
    spliced: bool = False          # True: follow the evasion; False: the raceline


def decide(now, plan, ego_received, max_plan_age, max_ego_age):
    """Whether /local_path is valid now. plan is a Plan or None; ego_received a time or None."""
    if plan is None:
        return Decision(False, 'no planner output yet')
    age = now - plan.received
    if age > max_plan_age:
        return Decision(False, f'planner output {age:.2f} s old')
    if ego_received is None:
        return Decision(False, 'no ego state yet')
    ego_age = now - ego_received
    if ego_age > max_ego_age:
        return Decision(False, f'ego state {ego_age:.2f} s old')
    if plan.status not in STATUSES:
        return Decision(False, f'unknown planner status {plan.status!r}')
    if plan.status not in COLLISION_FREE:
        return Decision(False, f'planner: {plan.status} ({plan.note})')
    if plan.status == EVADING and len(plan.s) == 0:
        return Decision(False, 'planner says evading but sent no trajectory')
    if plan.status == CLEAR and len(plan.s) != 0:
        return Decision(False, 'planner says clear but sent a trajectory')
    if plan.status == EVADING:
        return Decision(True, plan.note, spliced=True)
    return Decision(True, 'clear')


class Pairer:
    """Pairs trajectories with statuses by exact stamp; keeps the newest complete pair."""

    def __init__(self, keep=10):
        self.keep = keep
        self.trajectories = {}    # stamp -> (s, d)
        self.statuses = {}        # stamp -> (status, note)
        self.latest = None        # (stamp, Plan)

    def add_trajectory(self, stamp, s, d, now):
        self.trajectories[stamp] = (np.asarray(s, float), np.asarray(d, float))
        self._match(stamp, now)

    def add_status(self, stamp, status, note, now):
        self.statuses[stamp] = (status, note)
        self._match(stamp, now)

    def _match(self, stamp, now):
        if stamp in self.trajectories and stamp in self.statuses:
            if self.latest is None or stamp > self.latest[0]:
                (s, d), (status, note) = self.trajectories[stamp], self.statuses[stamp]
                self.latest = (stamp, Plan(status, note, s, d, now))
        for book in (self.trajectories, self.statuses):
            for old in sorted(book)[:-self.keep]:
                del book[old]

    @property
    def plan(self):
        return None if self.latest is None else self.latest[1]
