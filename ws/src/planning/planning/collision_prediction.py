"""Predict where the ego car will catch the opponent, as a Frenet region to avoid.

Port of collision_prediction.py, which is byte-identical in ForzaETH's Predictive
Spliner and ZJU's FSDP. Both comparison branches import this one copy, so any
difference between them comes from the planners and not from how each was ported.

Each cycle it takes the first dynamic obstacle as the opponent and rolls both cars
forward time_steps * dt seconds: the opponent at the speed its predicted trajectory
(/opponent_trajectory) gives for where it is, the ego at the speed
/global_waypoints_updated gives, accelerating towards that speed if it is below
it. The ego s where the gap first closes below save_distance_front starts the
region; the ego s where the ego is save_distance_back past the opponent ends it.
The region goes out as a single Obstacle the avoidance planner treats as blocked.

The behaviour is upstream's, including its rough edges, because changing it here
would make both branches differ from the code they claim to port:

  * Only the first dynamic obstacle is the opponent (upstream's own TODO).
  * The ego lap length is the s of the second-to-last updated waypoint, because
    the last waypoint is dropped as a duplicate of the first and the lap length
    is then read off what is left. Every wrap uses that, one waypoint spacing
    short of the real lap. The opponent lap length is the s of the last
    opponent waypoint, which is not dropped, though upstream's comment says it is.
  * Wrap handling at the start is a heuristic: ego past 2/3 of the lap and the
    opponent before 1/3 means the opponent is a lap ahead and gets one lap added.
  * The force-trailing check looks up the opponent's predicted d at that
    unwrapped position, so after the lap is added the nearest waypoint is the
    last one rather than the right one.
  * force_trailing is True when the opponent is more than 0.25 m off its
    predicted line, or fewer than one opponent lap has been observed.
  * A region that starts but does not end within the horizon ends at the
    opponent's final predicted s rather than the ego's, and is a wall from
    d = -10 to d = 10.
  * The region's s values are the raw rolled-forward positions, so they are not
    wrapped. s_start and s_center can be at or beyond the lap length, which
    breaks the convention in f1t_msgs/Obstacle.msg that s_center is in
    [0, lap_length).
  * The Obstacle's size and variance fields are left at 0 and is_visible False.
    Obstacle.msg asks for positive variances.
  * The expire counter counts loop cycles, not seconds. A region is deleted
    max_expire_counter cycles after the last one was found, and the empty
    array and marker deletes then go out every cycle.
  * Upstream reads speed_offset from the max_expire_counter parameter, but
    never uses speed_offset, so it is left out here.
  * The ego state defaults to s = 0, v = 0 until odometry arrives; upstream
    does not wait for it.

Upstream called the convert_frenet2globarr_service to place the two markers.
That is FrenetConverter.to_cartesian here, in-process, given the same s (taken
modulo the short ego lap length as above) and d.

No rclpy here, so it runs under plain pytest. collision_prediction_node.py is the wrapper.
"""

from dataclasses import dataclass

import numpy as np

TRAILING_D_TOLERANCE = 0.25    # m off the predicted line before we refuse to overtake
REGION_D_MARGIN = 0.25         # m added either side of the opponent's d
WALL_D = 10.0                  # m, a region with no end blocks the whole track


@dataclass
class CollisionParams:
    """Upstream's dynamic_reconfigure defaults."""

    time_steps: int = 200
    dt: float = 0.02                   # s
    save_distance_front: float = 0.6   # m
    save_distance_back: float = 0.4    # m
    max_v: float = 10.0                # m/s
    min_v: float = 0.0                 # m/s
    max_a: float = 5.5                 # m/s^2
    min_a: float = 5.0                 # m/s^2
    max_expire_counter: int = 10       # cycles


@dataclass
class Region:
    """The collision region, in the fields the node copies into an Obstacle."""

    s_start: float
    s_end: float
    s_center: float
    d_right: float
    d_left: float
    d_center: float
    begin_d: float     # opponent d at each end, for the markers
    end_d: float
    ended: bool        # False: no end within the horizon, so it is a wall


@dataclass
class Step:
    """What one cycle wants published. None means publish nothing on that topic."""

    force_trailing: object = None      # bool or None
    region: object = None              # Region or None
    delete: bool = False               # publish the empty array and marker deletes


def first_dynamic(obstacles):
    """The opponent is the first obstacle that is not static, or None."""
    for obs in obstacles:
        if not obs.is_static:
            return obs
    return None


def acceleration(v, p):
    """Linear in speed from min_a at max_v to max_a at min_v."""
    return p.min_a + (p.max_a - p.min_a) / max(p.max_v - p.min_v, 10e-6) * (p.max_v - v)


def predict_region(ego_s, ego_v, opp_s, opp_wpnts, ego_wpnts, max_s_opp, max_s_ego, p):
    """Roll both cars forward and return the Region, or None if they never meet.

    ego_s and opp_s are already wrap-adjusted as upstream does before the loop.
    """
    s_opp = np.array([w.s_m for w in opp_wpnts])
    s_ego = np.array([w.s_m for w in ego_wpnts])

    begin = end = False
    begin_s = end_s = 0.0
    begin_d = end_d = 0.0
    opp_d = 0.0
    v = ego_v

    for _ in range(p.time_steps):
        oi = int(np.abs(s_opp - opp_s % max_s_opp).argmin())
        ei = int(np.abs(s_ego - ego_s % max_s_ego).argmin())
        opp_v = opp_wpnts[oi].proj_vs_mps
        target_v = ego_wpnts[ei].vx_mps
        a = acceleration(v, p)

        if v < target_v:
            ego_s = ego_s + v * p.dt + 0.5 * a * p.dt ** 2
            opp_s = opp_s + opp_v * p.dt
            v += a * p.dt
        else:
            ego_s = ego_s + target_v * p.dt
            opp_s = opp_s + opp_v * p.dt

        opp_d = opp_wpnts[oi].d_m

        if (not begin
                and ((opp_s - ego_s) % max_s_ego < p.save_distance_front
                     or abs(opp_s - ego_s) < p.save_distance_front)):
            begin_s, begin_d, begin = ego_s, opp_d, True
        elif begin and not end and (ego_s - opp_s) > p.save_distance_back:
            end_s, end_d, end = ego_s, opp_d, True
            break

    if not begin:
        return None
    if not end:
        end_s, end_d = opp_s, opp_d

    if end:
        lo, hi = sorted((begin_d, end_d))
        d_right, d_left = lo - REGION_D_MARGIN, hi + REGION_D_MARGIN
    else:
        d_right, d_left = -WALL_D, WALL_D

    return Region(s_start=float(begin_s), s_end=float(end_s),
                  s_center=float((begin_s + end_s) / 2),
                  d_right=float(d_right), d_left=float(d_left),
                  d_center=float((d_right + d_left) / 2),
                  begin_d=float(begin_d), end_d=float(end_d), ended=end)


class CollisionPredictor:
    """The state upstream kept on its node. step() is one pass of its main loop."""

    def __init__(self, params=None):
        self.params = params or CollisionParams()
        self.opp_wpnts = []
        self.max_s_opp = None
        self.lap_count = None
        self.ego_wpnts = []
        self.max_s_ego = None
        self.opponent = None
        self.ego_s = 0.0
        self.ego_v = 0.0
        self.expire_counter = 0

    def set_opponent_trajectory(self, oppwpnts, lap_count):
        if not oppwpnts:
            return False    # upstream would raise IndexError here
        self.opp_wpnts = list(oppwpnts)
        self.max_s_opp = self.opp_wpnts[-1].s_m
        self.lap_count = lap_count
        return True

    def set_updated_waypoints(self, wpnts):
        trimmed = list(wpnts)[:-1]
        if not trimmed:
            return False    # upstream would raise IndexError here
        self.ego_wpnts = trimmed
        self.max_s_ego = trimmed[-1].s_m
        return True

    def set_obstacles(self, obstacles):
        self.opponent = first_dynamic(obstacles)

    def set_ego(self, s, v):
        self.ego_s, self.ego_v = float(s), float(v)

    def should_force_trailing(self, opp_s, opp_d):
        s_opp = np.array([w.s_m for w in self.opp_wpnts])
        idx = int(np.abs(s_opp - opp_s).argmin())
        return bool(abs(opp_d - self.opp_wpnts[idx].d_m) > TRAILING_D_TOLERANCE
                    or self.lap_count < 1)

    def step(self):
        out = Step()
        opp = self.opponent
        if opp is not None:
            ego_s = self.ego_s
            opp_s = opp.s_center
            if ego_s > self.max_s_ego * (2 / 3) and opp_s < self.max_s_ego * (1 / 3):
                opp_s += self.max_s_ego

            out.force_trailing = self.should_force_trailing(opp_s, opp.d_center)
            out.region = predict_region(ego_s, self.ego_v, opp_s, self.opp_wpnts,
                                        self.ego_wpnts, self.max_s_opp, self.max_s_ego,
                                        self.params)
            if out.region is not None:
                self.expire_counter = 0

        self.expire_counter += 1
        if self.expire_counter >= self.params.max_expire_counter:
            self.expire_counter = self.params.max_expire_counter
            out.delete = True
        return out

    def marker_positions(self, region, conv):
        """Map-frame (x, y) of the region's start and end, for the two markers."""
        bx, by = conv.to_cartesian(region.s_start % self.max_s_ego, region.begin_d)
        ex, ey = conv.to_cartesian(region.s_end % self.max_s_ego, region.end_d)
        return (float(bx), float(by)), (float(ex), float(ey))
