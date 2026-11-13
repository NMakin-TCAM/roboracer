"""Turn tracks/<name>/raceline.csv into the global waypoints the shared ports wait on.

The four shared ports (update_waypoints, collision_prediction, opponent_trajectory,
predictor_opponent_trajectory) all block until a WpntArray arrives on
/global_waypoints, and two of them also wait on /global_waypoints_scaled.
Upstream, ForzaETH's global planner publishes them. Here the raceline already
exists: build_raceline writes it to tracks/<name>/raceline.csv, and pure_pursuit,
obstacle_spliner_node and FrenetConverter.from_track all read it from there. This
module converts that file, field by field, into f1t_msgs/Wpnt. It imports no ROS;
global_waypoints_publisher_node.py publishes the result.

WHERE EACH Wpnt FIELD COMES FROM (Wpnt.msg, in order):

    id            row index
    s_m           raceline.csv s_m. The closing row (TUM repeats the first point at
                  s = lap length) is kept, because opponent_trajectory and the
                  predictor take wpnts[-1].s_m as the track length.
    d_m           0.0. The waypoints ARE the reference line: frenet_state_node
                  measures d against the raceline, so the raceline sits at d = 0 by
                  definition. Only reference=raceline is supported, for that reason.
    x_m, y_m      raceline.csv, map frame.
    d_right       distance from the raceline to the right track edge, positive, as
    d_left        upstream's (opponent_trajectory compares abs(d) with d_right).
                  Computed: centerline.csv carries widths about the centerline, and
                  FrenetConverter.load_bounds re-expresses both edges about the
                  raceline. raceline.csv has no widths of its own.
    psi_rad       ROS yaw: 0 = +x, counter-clockwise. raceline.csv holds TUM's
                  convention, yaw - pi/2 (raceline.py), so pi/2 is added back. The
                  file's convention is checked against the geometry before use, by
                  frenet.yaw_from_csv_psi, the check FrenetConverter.from_raceline uses.
                  ForzaETH's global planner converts TUM headings the same way
                  before publishing. None of the four ports reads psi_rad today.
    kappa_radpm   raceline.csv. Recomputed from the geometry (raceline.path_traj's
                  three-point circle) only if the column is zero or non-finite.
    vx_mps        raceline.csv. A raceline without positive speeds is refused: the
                  speed is the one thing the ports cannot do without (update_waypoints
                  and the predictor divide by the top speed) and it cannot be honestly
                  invented here. Re-run build_raceline.
    ax_mps2       raceline.csv. Recomputed as (v1^2 - v0^2) / (2 ds), the formula of
                  raceline.speed_profile, only if the column is all zero while the
                  speeds vary.

Everything recomputed or suspicious is returned in `warnings` so the node can log
it; nothing is filled with zeros silently.
"""

import dataclasses
import math
import os

import numpy as np

from planning.frenet import FrenetConverter, yaw_from_csv_psi
from planning.raceline import path_traj, read_provenance, read_raceline

# raceline.csv columns: s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
_S, _X, _Y, _PSI, _KAPPA, _VX, _AX = range(7)

# Fields and order of f1t_msgs/Wpnt. test_global_waypoints_publisher.py checks
# this against Wpnt.msg itself, so the two cannot drift apart.
WPNT_FIELDS = ('id', 's_m', 'd_m', 'x_m', 'y_m', 'd_right', 'd_left',
               'psi_rad', 'kappa_radpm', 'vx_mps', 'ax_mps2')

@dataclasses.dataclass(frozen=True)
class Waypoint:
    id: int
    s_m: float
    d_m: float
    x_m: float
    y_m: float
    d_right: float
    d_left: float
    psi_rad: float
    kappa_radpm: float
    vx_mps: float
    ax_mps2: float


@dataclasses.dataclass
class GlobalWaypoints:
    waypoints: list
    warnings: list
    provenance: dict = None

    @property
    def lap_length(self):
        return self.waypoints[-1].s_m


def load_global_waypoints(track_dir):
    """Read tracks/<name>/raceline.csv (+ centerline.csv for widths) into Waypoints."""
    raceline_csv = os.path.join(track_dir, 'raceline.csv')
    centerline_csv = os.path.join(track_dir, 'centerline.csv')
    if not os.path.isfile(raceline_csv):
        raise FileNotFoundError(f'{raceline_csv} not found; run build_raceline first')
    if not os.path.isfile(centerline_csv):
        raise FileNotFoundError(
            f'{centerline_csv} not found. Wpnt.d_left/d_right are the track widths and '
            f'raceline.csv has none; they are computed from the centerline widths')

    t = read_raceline(raceline_csv)
    if t.shape[1] != 7:
        raise ValueError(f'{raceline_csv}: expected 7 columns, got {t.shape[1]}')
    if not np.all(np.isfinite(t[:, [_S, _X, _Y, _PSI]])):
        raise ValueError(f'{raceline_csv}: non-finite s, x, y or psi')
    warnings = []

    # Closed: the last row repeats the first point at s = lap length. Keep it closed.
    if not np.allclose(t[0, [_X, _Y]], t[-1, [_X, _Y]], atol=1e-6):
        close = t[0].copy()
        close[_S] = t[-1, _S] + math.hypot(*(t[0, [_X, _Y]] - t[-1, [_X, _Y]]))
        t = np.vstack((t, close))
        warnings.append('raceline.csv is not closed (TUM repeats the first point as the '
                        'last row); appended the closing row, so wpnts[-1].s_m is the lap '
                        'length the ports take it to be.')
    if np.any(np.diff(t[:, _S]) <= 0):
        raise ValueError(f'{raceline_csv}: s_m is not strictly increasing')

    open_t = t[:-1]                      # one row per distinct point
    xy = open_t[:, [_X, _Y]]

    yaw, warning = yaw_from_csv_psi(open_t[:, _PSI], xy)
    if warning:
        warnings.append(warning)

    kappa = open_t[:, _KAPPA]
    if not np.all(np.isfinite(kappa)) or np.max(np.abs(np.nan_to_num(kappa))) < 1e-9:
        kappa = path_traj(xy)[:-1, 4]
        warnings.append('raceline.csv kappa_radpm is zero or non-finite; computed from '
                        'the geometry (three-point circle over +-0.4 m).')

    vx = open_t[:, _VX]
    if not np.all(np.isfinite(vx)) or np.any(vx < 0) or np.max(vx) <= 0:
        raise ValueError(
            f'{raceline_csv}: vx_mps is not a usable speed profile ({np.sum(~np.isfinite(vx))} '
            f'non-finite, {np.sum(vx < 0)} negative, {np.sum(vx > 0)} positive of {len(vx)}). '
            f'The speeds cannot be invented here; re-run build_raceline.')

    ax = open_t[:, _AX]
    if not np.all(np.isfinite(ax)) or (np.max(np.abs(np.nan_to_num(ax))) < 1e-9
                                       and np.ptp(vx) > 1e-6):
        ds = np.diff(t[:, _S])
        ax = (np.roll(vx, -1) ** 2 - vx ** 2) / (2 * ds)
        warnings.append('raceline.csv ax_mps2 is zero although the speeds vary; computed '
                        'as (v1^2 - v0^2) / (2 ds).')

    conv = FrenetConverter(xy, s=open_t[:, _S], psi=yaw, kappa=kappa, vx=vx)
    conv.load_bounds(centerline_csv)
    d_left, d_right = conv.d_left, -conv.d_right         # both positive, as upstream's
    outside = int(np.sum((d_left <= 0) | (d_right <= 0)))
    if outside:
        warnings.append(f'{outside} of {len(d_left)} raceline points lie outside the track '
                        f'edges from centerline.csv (a width <= 0). Every consumer that '
                        f'tests d against these bounds will misbehave there.')

    def closed(a):
        return np.append(a, a[0])

    cols = {
        's_m': t[:, _S], 'd_m': np.zeros(len(t)),
        'x_m': t[:, _X], 'y_m': t[:, _Y],
        'd_right': closed(d_right), 'd_left': closed(d_left),
        'psi_rad': closed(yaw), 'kappa_radpm': closed(kappa),
        'vx_mps': closed(vx), 'ax_mps2': closed(ax),
    }
    waypoints = [Waypoint(id=i, **{k: float(v[i]) for k, v in cols.items()})
                 for i in range(len(t))]
    return GlobalWaypoints(waypoints, warnings, read_provenance(raceline_csv))


def provenance_warning(provenance):
    """A warning for anything but a clean mintime raceline, as pure_pursuit_node gives."""
    if provenance is None:
        return 'raceline.csv has no provenance header (older or hand-made file).'
    if provenance.get('method') != 'mintime' or not provenance.get('clean', False):
        return (f'raceline.csv is not a clean mintime raceline: method '
                f'{provenance.get("method")!r}, clean {provenance.get("clean")}, notes '
                f'{provenance.get("notes")}')
    return None
