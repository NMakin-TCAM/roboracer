"""Reactive overtaking: pick a side, spline around the obstacle, rejoin the raceline.

This is the non-predictive baseline. It assumes an obstacle stays where it was
last seen, which is wrong for a moving opponent and is exactly the assumption
Predictive Spliner replaces with a Gaussian process over the opponent's past
laps. Having the baseline matters for the comparison: it is the floor that
prediction has to beat, and it is most of the scaffolding prediction plugs into.

No rclpy here, so it runs under plain pytest. spliner_node.py is the wrapper.

Everything works in arc length measured FORWARD FROM THE CAR, written u, with
u = ds(ego_s, s). That sidesteps the start/finish line entirely: u is monotonic
over the manoeuvre whether or not it crosses s = 0, and the wrap happens once at
the end when u is converted back to s.

THE PATH (build_evasion). Three quintic smoothsteps in d over u: from the car's d
to the apex, the apex held across the obstacle, and from the apex back to the
raceline (d = 0). The smoothstep S(t) = 10t^3 - 15t^4 + 6t^5 is monotone on
[0, 1] with zero slope and curvature at both ends, so:

  * d only ever moves from the car's d TOWARD the apex. It cannot start the wrong
    way. The five-knot cubic spline this replaces bulged up to 0.3 m away from the
    chosen side at the start, because a cubic spline couples every knot to every
    other; with replanning, the car executed nothing but those wrong-way starts.
  * curvature is continuous, so pure pursuit gets no kink, and each piece depends
    only on its own two ends.
  * it is the easing obstacle_spliner_node uses, so the two planners shape paths
    alike.

The car's footprint sets where the apex must hold: ego s is the rear axle
(base_link), the nose is car_front ahead of it and the tail car_rear behind. The
full offset is reached apex_margin before the nose reaches the obstacle and held
until the tail is apex_margin past it. A transition may be no steeper than
max_lateral_slope (dd/ds; S peaks at 1.875 x the mean slope); if the car is too
close for that, the status is TOO_CLOSE.

COMMITMENT (Spliner). plan() derives a path from the car's current state. Called
every cycle it re-derives the lead-in each time, and the lead-in shrinks to
nothing as the car approaches: the replay found it giving up 0.25 m short of the
obstacle. Spliner instead keeps the path it started, fixed in absolute s, and
reports what is left of it from the car on. It releases the commitment, and plans
afresh, on exactly these conditions (Spliner.release_reason, checked in order):

  passed        the car is past the end of the committed path
  gone          no obstacle in the current array matches the committed one, while
                it is still ahead of the car's nose (once alongside, a forward
                sensor loses it; that is held until 'passed')
  blocked       the rest of the committed path is no longer collision-free
                against the current obstacles (path_clear)
  off line      the car is more than max_commit_offset from the committed d

STATUS. plan() returns a PlanResult whose status says which case it is, because
an empty trajectory means two opposite things: "nothing in the way, follow the
raceline" (CLEAR) and "something is in the way and I cannot get round it"
(NO_ROOM, TOO_CLOSE). A consumer that read the empty list as "follow the
raceline" would drive into an obstacle the planner had refused. Branch on
status, never on len(s). collision_free is True only for CLEAR and EVADING.
"""

from dataclasses import dataclass, field

import numpy as np

LEFT, RIGHT = 'left', 'right'

CLEAR = 'clear'          # nothing in the way: follow the raceline
EVADING = 'evading'      # s, d, v is the way round
NO_ROOM = 'no room'      # in the way, and neither side has room: do not drive on
TOO_CLOSE = 'too close'  # in the way, and too close to spline round: do not drive on
STATUSES = (CLEAR, EVADING, NO_ROOM, TOO_CLOSE)
COLLISION_FREE = (CLEAR, EVADING)


@dataclass
class SplinerParams:
    """Tuning. Defaults are deliberately timid; a baseline that crashes is not a baseline."""

    lookahead: float = 10.0          # only plan around obstacles within this many metres
    on_line_threshold: float = 1.0   # |d_center| beyond this and it is not in our way
    car_width: float = 0.30
    evasion_dist: float = 0.35       # clearance between car edge and obstacle edge at the apex
    bound_margin: float = 0.15       # never plan closer than this to the track edge
    back_to_raceline_after: float = 2.0    # metres of rejoin after it, at least
    car_front: float = 0.45          # m, rear axle (base_link, ego s) to the nose
    car_rear: float = 0.10           # m, rear axle to the tail
    apex_margin: float = 0.15        # m: full offset this far before the nose meets it and
                                     # after the tail leaves it
    max_lateral_slope: float = 0.5   # dd/ds at the steepest point of a transition
    max_commit_offset: float = 0.30  # m off the committed line before it is abandoned
    match_tolerance: float = 0.5     # m an obstacle may move and still be the committed one
    speed_scale: float = 0.8         # fraction of raceline speed through the manoeuvre
    resolution: float = 0.10         # spacing of published waypoints, metres
    switch_hysteresis: float = 0.25  # the other side must be this much roomier to flip


@dataclass
class SideChoice:
    side: str or None
    d_apex: float
    space_left: float
    space_right: float
    reason: str


def obstacles_ahead(obstacles, ego_s, conv, params):
    """Obstacles in our way, nearest first.

    'In our way' means ahead of us, inside the lookahead, and laterally close
    enough to the reference line to matter. Comparison uses signed lap distance,
    never subtraction, so a car just past the start line is correctly seen as
    ahead rather than a lap behind.
    """
    out = []
    for o in obstacles:
        gap = conv.ds(ego_s, o.s_center)
        if not (0.0 < gap < params.lookahead):
            continue
        if abs(o.d_center) > params.on_line_threshold:
            continue
        out.append((gap, o))
    out.sort(key=lambda p: p[0])
    return [o for _, o in out]


def choose_side(obs, conv, params, current_side=None):
    """Which way round. Returns a SideChoice; side is None when neither fits.

    Space is measured against the real track bounds at the obstacle, not against
    a fixed corridor, so the planner refuses to squeeze through a gap that only
    exists on a wide part of the track.
    """
    d_left_track, d_right_track = conv.width_at(obs.s_center)
    d_left_track = float(d_left_track) - params.bound_margin
    d_right_track = float(d_right_track) + params.bound_margin

    needed = params.car_width + params.evasion_dist
    space_left = d_left_track - obs.d_left
    space_right = obs.d_right - d_right_track

    left_ok = space_left >= needed
    right_ok = space_right >= needed

    if not left_ok and not right_ok:
        return SideChoice(None, 0.0, space_left, space_right, 'no feasible side')

    if left_ok and right_ok:
        # Both work, so prefer the roomier one, but make switching cost something.
        if current_side == LEFT and space_right - space_left < params.switch_hysteresis:
            side = LEFT
        elif current_side == RIGHT and space_left - space_right < params.switch_hysteresis:
            side = RIGHT
        else:
            side = LEFT if space_left >= space_right else RIGHT
        reason = 'both sides clear'
    else:
        side = LEFT if left_ok else RIGHT
        reason = 'only side with room'

    if side == LEFT:
        d_apex = min(obs.d_left + params.car_width / 2.0 + params.evasion_dist, d_left_track)
    else:
        d_apex = max(obs.d_right - params.car_width / 2.0 - params.evasion_dist, d_right_track)

    return SideChoice(side, float(d_apex), space_left, space_right, reason)


def smoothstep(t):
    """Quintic smoothstep: 0 -> 1 on [0, 1], monotone, zero slope and curvature at both ends."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


SMOOTHSTEP_PEAK_SLOPE = 1.875   # max of S'(t), at t = 1/2


def transition_length(delta_d, params):
    """Shortest distance over which a smoothstep moves delta_d within max_lateral_slope."""
    return SMOOTHSTEP_PEAK_SLOPE * abs(delta_d) / params.max_lateral_slope


def arc(s, lap):
    """Arc length along s from its first sample, through the wrap."""
    return np.concatenate(([0.0], np.cumsum(np.diff(s) % lap)))


def build_evasion(ego_s, ego_d, obs, side_choice, conv, params):
    """Sample the three-smoothstep path (see THE PATH). Returns (s, d, v) arrays, or None.

    None means too close: the apex cannot be reached before the nose meets the obstacle
    without a transition steeper than max_lateral_slope.
    """
    if side_choice.side is None:
        return None
    apex = side_choice.d_apex

    u_obs_start = conv.ds(ego_s, obs.s_start)
    u_obs_end = u_obs_start + max(conv.ds(obs.s_start, obs.s_end), 0.0)
    u_reach = u_obs_start - params.car_front - params.apex_margin   # full offset by here
    u_leave = u_obs_end + params.car_rear + params.apex_margin      # held until here
    if u_reach <= 0.0 or u_reach < transition_length(apex - ego_d, params):
        return None
    u_end = u_leave + max(params.back_to_raceline_after, transition_length(apex, params))

    n = max(int(np.ceil(u_end / params.resolution)) + 1, 2)
    u = np.linspace(0.0, u_end, n)
    d = np.where(u <= u_reach, ego_d + (apex - ego_d) * smoothstep(u / u_reach),
                 np.where(u <= u_leave, apex,
                          apex * (1.0 - smoothstep((u - u_leave) / (u_end - u_leave)))))

    # Clamp to the track. choose_side already keeps the apex inside, so this only bites if
    # the car starts outside the margin.
    s = np.mod(ego_s + u, conv.length)
    hi, lo = conv.width_at(s)
    d = np.clip(d, np.asarray(lo) + params.bound_margin, np.asarray(hi) - params.bound_margin)

    v = np.asarray(conv.speed_at(s), dtype=float) * params.speed_scale
    return s, d, v


def path_clear(s, d, obstacles, conv, params):
    """True when the path (s, d) keeps clear of every obstacle.

    Wherever the car's footprint along the path (car_rear behind to car_front ahead of a
    sample) overlaps an obstacle's s span, the car's near edge must be at least
    evasion_dist / 2 beyond the obstacle's edge: half the planning clearance, so a committed
    path is not dropped for the noise of a re-detection, but is dropped for a real change.
    """
    if len(s) == 0:
        return True
    c = params.car_width / 2.0 + params.evasion_dist / 2.0
    u = arc(s, conv.length)
    for o in obstacles:
        lo = conv.ds(s[0], o.s_start) - params.car_front
        hi = conv.ds(s[0], o.s_end) + params.car_rear
        near = (u >= lo) & (u <= hi)
        if not np.any(near):
            continue
        dn = d[near]
        if not np.all((dn >= o.d_left + c) | (dn <= o.d_right - c)):
            return False
    return True


def _empty():
    return np.array([])


@dataclass
class PlanResult:
    """One planning cycle's answer. s, d, v are non-empty exactly when status is EVADING."""

    status: str
    note: str
    side: str = None
    obstacle: object = None          # the obstacle planned around or refused, None when CLEAR
    s: np.ndarray = field(default_factory=_empty)
    d: np.ndarray = field(default_factory=_empty)
    v: np.ndarray = field(default_factory=_empty)
    committed: bool = False          # Spliner: continuing an evasion started earlier
    released: str = None             # Spliner: the release condition met this cycle, if any

    @property
    def collision_free(self):
        """True when driving on is fine: the raceline (CLEAR) or the evasion (EVADING)."""
        return self.status in COLLISION_FREE


def plan(ego_s, ego_d, obstacles, conv, params, current_side=None):
    """One planning cycle from the car's current state, with no memory. See STATUS.

    Drive a car with Spliner: called every cycle this re-derives the lead-in from the car
    and gives up as the car closes in (see COMMITMENT).
    """
    ahead = obstacles_ahead(obstacles, ego_s, conv, params)
    if not ahead:
        return PlanResult(CLEAR, 'clear')

    obs = ahead[0]
    choice = choose_side(obs, conv, params, current_side)
    if choice.side is None:
        return PlanResult(NO_ROOM, f'trailing: {choice.reason}', obstacle=obs)

    built = build_evasion(ego_s, ego_d, obs, choice, conv, params)
    if built is None:
        return PlanResult(TOO_CLOSE, 'too close to plan an evasion', obstacle=obs)

    s, d, v = built
    if not path_clear(s, d, obstacles, conv, params):
        return PlanResult(NO_ROOM, f'trailing: the way {choice.side} is blocked by another '
                                   f'obstacle', obstacle=obs)
    return PlanResult(EVADING, f'overtaking {choice.side} ({choice.reason})',
                      side=choice.side, obstacle=obs, s=s, d=d, v=v)


@dataclass
class Commitment:
    """An evasion in progress, fixed in absolute s from when it started."""

    s: np.ndarray
    d: np.ndarray
    v: np.ndarray
    side: str
    obstacle: object
    note: str


class Spliner:
    """plan() with memory: commits to an evasion once it starts. See COMMITMENT.

    step() returns a PlanResult. result.committed is True while continuing an evasion
    started earlier; result.released names the release condition met this cycle (passed,
    gone, blocked, off line), else None.
    """

    def __init__(self, conv, params):
        self.conv = conv
        self.params = params
        self.committed = None
        self.side = None            # last side taken, for choose_side's hysteresis

    def _remaining(self, ego_s):
        """Index of the first committed sample at or ahead of the car, or None past the end."""
        c = self.committed
        u = arc(c.s, self.conv.length)
        u_car = self.conv.ds(c.s[0], ego_s)
        if u_car > u[-1]:
            return None
        return min(int(np.searchsorted(u, max(u_car, 0.0), side='left')), len(u) - 1)

    def _matches(self, o):
        c = self.committed.obstacle
        tol = self.params.match_tolerance
        return (abs(self.conv.ds(c.s_center, o.s_center)) <= tol
                and abs(o.d_center - c.d_center) <= tol)

    def release_reason(self, ego_s, ego_d, obstacles):
        """Why the commitment must go now, or None to keep it. See COMMITMENT."""
        c = self.committed
        i = self._remaining(ego_s)
        if i is None:
            return 'passed'
        nose_reached_it = self.conv.ds(ego_s + self.params.car_front, c.obstacle.s_start) <= 0
        if not nose_reached_it and not any(self._matches(o) for o in obstacles):
            # Only while it is still ahead of the nose: a forward sensor loses an obstacle
            # once it is beside or behind the car, and releasing then would steer the car
            # back into it. Alongside, its disappearance is held until 'passed'.
            return 'gone'
        if not path_clear(c.s[i:], c.d[i:], obstacles, self.conv, self.params):
            return 'blocked'
        u = arc(c.s, self.conv.length)
        d_here = float(np.interp(max(self.conv.ds(c.s[0], ego_s), 0.0), u, c.d))
        if abs(ego_d - d_here) > self.params.max_commit_offset:
            return 'off line'
        return None

    def step(self, ego_s, ego_d, obstacles):
        released = None
        if self.committed is not None:
            released = self.release_reason(ego_s, ego_d, obstacles)
            if released is None:
                c = self.committed
                i = self._remaining(ego_s)
                return PlanResult(EVADING, c.note, side=c.side, obstacle=c.obstacle,
                                  s=c.s[i:], d=c.d[i:], v=c.v[i:], committed=True)
            self.committed = None

        result = plan(ego_s, ego_d, obstacles, self.conv, self.params, self.side)
        result.released = released
        if result.status == EVADING:
            self.committed = Commitment(result.s, result.d, result.v, result.side,
                                        result.obstacle, result.note)
            self.side = result.side
        return result
