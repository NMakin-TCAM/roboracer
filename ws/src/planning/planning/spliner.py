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
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

LEFT, RIGHT = 'left', 'right'


@dataclass
class SplinerParams:
    """Tuning. Defaults are deliberately timid; a baseline that crashes is not a baseline."""

    lookahead: float = 10.0          # only plan around obstacles within this many metres
    on_line_threshold: float = 1.0   # |d_center| beyond this and it is not in our way
    car_width: float = 0.30
    evasion_dist: float = 0.35       # clearance between car edge and obstacle edge at the apex
    bound_margin: float = 0.15       # never plan closer than this to the track edge
    back_to_raceline_before: float = 1.5   # metres of lead-in before the obstacle
    back_to_raceline_after: float = 2.0    # metres of rejoin after it
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


def build_evasion(ego_s, ego_d, obs, side_choice, conv, params):
    """Sample a cubic through the manoeuvre. Returns (s, d, v) arrays, or None.

    Knots: where we are now, the lead-in, the apex held across the obstacle, and
    the rejoin. A cubic through those gives continuous curvature, which matters
    because a path with a kink in it is one the controller cannot follow.
    """
    if side_choice.side is None:
        return None

    u_start = 0.0
    u_lead = max(conv.ds(ego_s, obs.s_start) - params.back_to_raceline_before, 0.3)
    u_apex_in = conv.ds(ego_s, obs.s_start)
    u_apex_out = u_apex_in + max(conv.ds(obs.s_start, obs.s_end), 0.1)
    u_end = u_apex_out + params.back_to_raceline_after

    knots_u = [u_start, u_lead, u_apex_in, u_apex_out, u_end]
    knots_d = [ego_d, ego_d * 0.5, side_choice.d_apex, side_choice.d_apex, 0.0]

    # A cubic needs strictly increasing knots. If the obstacle is already on top
    # of us the lead-in collapses, and there is no room to plan anything useful.
    for a, b in zip(knots_u, knots_u[1:]):
        if b - a < 1e-3:
            return None

    spline = CubicSpline(knots_u, knots_d, bc_type=((1, 0.0), (1, 0.0)))

    n = max(int(u_end / params.resolution), 2)
    u = np.linspace(0.0, u_end, n)
    d = spline(u)

    # Clamp to the track. Clamping after splining can introduce a kink, so a
    # clamp that actually bites is a sign the apex was too ambitious; the node
    # logs it rather than silently publishing a path through the wall.
    s = np.mod(ego_s + u, conv.length)
    hi, lo = conv.width_at(s)
    d = np.clip(d, np.asarray(lo) + params.bound_margin, np.asarray(hi) - params.bound_margin)

    v = np.asarray(conv.speed_at(s), dtype=float) * params.speed_scale

    return s, d, v


def plan(ego_s, ego_d, obstacles, conv, params, current_side=None):
    """One planning cycle. Returns (s, d, v, side, note).

    side is None and the arrays are empty whenever we are not overtaking, which
    the controller reads as 'track the global raceline'.
    """
    empty = (np.array([]), np.array([]), np.array([]))

    ahead = obstacles_ahead(obstacles, ego_s, conv, params)
    if not ahead:
        return (*empty, None, 'clear')

    obs = ahead[0]
    choice = choose_side(obs, conv, params, current_side)
    if choice.side is None:
        return (*empty, None, f'trailing: {choice.reason}')

    built = build_evasion(ego_s, ego_d, obs, choice, conv, params)
    if built is None:
        return (*empty, None, 'too close to plan an evasion')

    s, d, v = built
    return s, d, v, choice.side, f'overtaking {choice.side} ({choice.reason})'
