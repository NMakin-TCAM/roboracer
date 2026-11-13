"""Predict how the opponent returns to its learned line after leaving it.

Port of predictor_opponent_trajectory.py, which is byte-identical in ForzaETH's
Predictive Spliner and ZJU's FSDP. Both comparison branches import this one
copy, so any difference between them comes from the planners and their main
GPs, not from this stage.

It acts only on the half laps opponent_trajectory.py publishes with
opp_is_on_trajectory False: more than 5 consecutive detections off the
predicted line. It fits a Gaussian process in d over the last 15 detections
plus the next 15 points of the stored /opponent_trajectory that are more than
7 m beyond them. It writes the fitted d (and resampled vd) into the stored
trajectory's waypoints across that span, and publishes the result back on
/opponent_trajectory. That message is also what ends opponent_trajectory's 3 s
wait.

The behaviour is upstream's, including its rough edges, because changing it
here would make both branches differ from the code they claim to port:

  * It publishes on /opponent_trajectory, the topic it subscribes to and the
    main GP publishes on. Its own output comes back as the stored trajectory,
    so a second off-trajectory prediction builds on the first, until the GP
    publishes again. The stored waypoints are also edited in place.
  * Only d is predicted. proj_vs_mps is left as the stored trajectory had it
    (upstream's resampling of vs is commented out), so the predicted return
    path keeps the old speed profile. vd is linearly interpolated from the
    merged points, which assumes their s is increasing; np.interp is silently
    wrong otherwise.
  * Predicted waypoints get d_var = the GP's standard deviation (not a
    variance) and vs_var = 69, a sentinel the markers use to colour them red.
    The main GP fills vs_var with its sigma_vs, so anything reading vs_var as
    a variance gets 69 m^2/s^2 on predicted points.
  * A waypoint is replaced only when its s is within 1e-8 of a /global_waypoints
    s in the fitted span (taken modulo the track length). The GP's waypoints
    are laid on those s, so this holds for them, and other points are left as
    they were.
  * The 15 trajectory points are the first 15 in list order beyond the last
    detection + 7 m, over the trajectory repeated for a second lap. Near the
    end of the second lap fewer or none are found, and the GP is fitted to
    detections alone.
  * Detections are unwrapped by adding a lap to every detection from the
    first consecutive pair more than half a lap apart onwards. The comparison
    uses the already-shifted s, and the shift edits the received message.
  * The second-lap copies of the trajectory carry s, d, x, y, proj_vs and vd,
    but d_var and vs_var are left at 0.
  * Two track lengths: the double-lap s list is fixed from the first
    /global_waypoints, the predictions use the latest one.
  * The GP kernel is C(0.5) * Matern(1.0, nu=1.5) + C(0.2) * White(1.0), with
    constants bounded in [1e-6, 1e3], fitted with one L-BFGS-B run from those
    values and no restarts. It is deterministic.
  * On-trajectory half laps and half laps with no detections are ignored.

Where upstream would crash it is made not to, as in the other ports:

  * sklearn before 1.1 (1.0.2, and Ubuntu 22.04's apt python3-sklearn 0.23.2)
    returns the GP mean as (n, 1). Upstream passes that to ForzaETH's
    FrenetConverter.get_cartesian, whose in-place `x += d * cos(...)` then
    raises ValueError on every prediction, so upstream never publishes
    (checked against ForzaETH's ROS 1 converter). From 1.1 the mean is (n,).
    ForzaETH's requirements leave scikit-learn unpinned, so their runs had a
    working version. The port flattens the mean, and with sklearn 1.1 or later
    it agrees with upstream exactly. This is why the image needs pip
    scikit-learn (1.3.2, as the TUM venv pins), not the apt package.
  * Only the two constant terms are bounded. The Matern length scale and the
    white-noise level keep sklearn's default bounds (1e-5, 1e5), and fits do
    run into them (sklearn warns).
  * If no /global_waypoints s lies inside the fitted span, upstream's predict
    raises ValueError on an empty array. The port skips that half lap.
  * If every global waypoint speed is 0, upstream's marker heights divide by
    zero. The port draws them with height 0.

No rclpy here, so it runs under plain pytest. predictor_opponent_trajectory_node.py
is the wrapper.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

NR_OF_POINTS_TO_MERGE = 15   # detections kept, and trajectory points added
GAP = 7.0                    # m beyond the last detection before trajectory points count
MATCH_S = 1e-8               # m, a waypoint is replaced if its s is this close
PREDICTED = 69.0             # vs_var sentinel for predicted waypoints


def make_kernel():
    """Upstream's kernel for d."""
    c1 = ConstantKernel(constant_value=0.5, constant_value_bounds=(1e-6, 1e3))
    c2 = ConstantKernel(constant_value=0.2, constant_value_bounds=(1e-6, 1e3))
    return c1 * Matern(length_scale=1.0, nu=3 / 2) + c2 * WhiteKernel(noise_level=1)


def _optimizer(obj_func, initial_theta, bounds):
    solution, function_value, _ = fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds)
    return solution, function_value


@dataclass
class ShiftedWpnt:
    """A trajectory waypoint copied one lap on, with the fields upstream copied."""

    s_m: float
    d_m: float
    x_m: float
    y_m: float
    proj_vs_mps: float
    vd_mps: float
    d_var: float = 0.0
    vs_var: float = 0.0


@dataclass
class MarkerRec:
    """One /opponent_traj_markerarray cylinder. Position is already map-frame."""

    id: int
    x: float
    y: float
    height: float
    width: float
    predicted: bool


@dataclass
class Prediction:
    """What one off-trajectory half lap publishes."""

    lap_count: float
    oppwpnts: list           # the stored trajectory's waypoints, edited in place
    markers: list


def double_lap(ego_s, track_length):
    """Upstream drops the duplicated last waypoint, then appends the s list one lap on."""
    original = list(ego_s)[:-1]
    return original + [s + track_length for s in original]


def double_trajectory(oppwpnts, track_length):
    """The trajectory followed by a copy one lap on (same objects first, then copies)."""
    doublelap = list(oppwpnts)
    for w in oppwpnts:
        doublelap.append(ShiftedWpnt(s_m=w.s_m + track_length, d_m=w.d_m, x_m=w.x_m,
                                     y_m=w.y_m, proj_vs_mps=w.proj_vs_mps, vd_mps=w.vd_mps))
    return doublelap


def unwrap_detections(detections, track_length):
    """From the first consecutive pair more than half a lap apart, add a lap to every later one.

    Edits the detections in place, and compares against already-shifted s.
    """
    around_origin = False
    for i in range(len(detections) - 1):
        if abs(detections[i].s - detections[i + 1].s) > track_length / 2 or around_origin:
            detections[i + 1].s = detections[i + 1].s + track_length
            around_origin = True


def merge_points(detections, doublelap):
    """The GP's training data: the last 15 detections, then the first 15 trajectory
    points in list order lying more than GAP beyond the last detection.

    Returns (s, d, vd) lists.
    """
    n = len(detections)
    last_s = detections[-1].s
    first = n - min(NR_OF_POINTS_TO_MERGE, n)
    opp_s = [detections[i].s for i in range(first, n)]
    opp_d = [detections[i].d for i in range(first, n)]
    opp_vd = [detections[i].vd for i in range(first, n)]
    counter = 0
    for w in doublelap:
        if w.s_m > last_s + GAP and counter < NR_OF_POINTS_TO_MERGE:
            opp_s.append(w.s_m)
            opp_d.append(w.d_m)
            opp_vd.append(w.vd_mps)
            counter += 1
    return opp_s, opp_d, opp_vd


def predict_return(oppwpnts, detections, ego_s_doublelap, track_length, kernel, to_xy):
    """Fit the merged points and write the prediction into oppwpnts. Returns False if skipped.

    to_xy(s_list, d_array) -> (xs, ys) places the predicted points in the map.
    detections are edited (unwrapped) in place, as upstream did.
    """
    doublelap = double_trajectory(oppwpnts, track_length)
    unwrap_detections(detections, track_length)
    opp_s, opp_d, opp_vd = merge_points(detections, doublelap)
    last_pred, first_pred = opp_s[-1], opp_s[0]

    s_pred = [s for s in ego_s_doublelap if first_pred <= s <= last_pred]

    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=_optimizer)
    gpr.fit(np.array(opp_s).reshape(-1, 1), np.array(opp_d).reshape(-1, 1))
    if not s_pred:
        return False        # upstream would raise ValueError in predict
    d_pred, sigma_d = gpr.predict(np.array(s_pred).reshape(-1, 1), return_std=True)
    d_pred = np.ravel(d_pred)       # (n, 1) before sklearn 1.x; upstream crashed on that
    vd_pred = np.interp(s_pred, opp_s, opp_vd)

    ego_s = [s % track_length for s in s_pred]
    xs, ys = to_xy(ego_s, d_pred)

    for w in oppwpnts:
        for j in range(len(ego_s)):
            if abs(ego_s[j] - w.s_m) < MATCH_S:
                w.x_m = xs[j]
                w.y_m = ys[j]
                w.d_m = d_pred[j]
                w.vd_mps = vd_pred[j]
                w.d_var = sigma_d[j]
                w.vs_var = PREDICTED
    return True


def marker_records(oppwpnts, max_velocity):
    """Upstream's cylinders: height is speed over the top speed, width follows d_var."""
    out = []
    for i, w in enumerate(oppwpnts):
        height = w.proj_vs_mps / max_velocity if max_velocity else 0.0
        width = min(max(5 * w.d_var, 0.07), 0.7)
        out.append(MarkerRec(id=i, x=w.x_m, y=w.y_m, height=height, width=width,
                             predicted=w.vs_var == PREDICTED))
    return out


class ReturnPredictor:
    """The state upstream kept on its node. on_proj_trajectory() is one loop pass."""

    def __init__(self):
        self.kernel = make_kernel()
        self.track_length = None        # latest /global_waypoints
        self.ego_s_doublelap = None     # fixed from the first
        self.max_velocity = None
        self.oppwpnts = None            # the stored /opponent_trajectory waypoints

    @property
    def ready(self):
        return self.ego_s_doublelap is not None and self.oppwpnts is not None

    def set_global_waypoints(self, wpnts):
        wpnts = list(wpnts)
        if not wpnts:
            return False    # upstream would raise IndexError here
        self.track_length = wpnts[-1].s_m
        if self.ego_s_doublelap is None:
            self.max_velocity = max(w.vx_mps for w in wpnts)
            self.ego_s_doublelap = double_lap([w.s_m for w in wpnts], self.track_length)
        return True

    def set_opponent_trajectory(self, oppwpnts):
        self.oppwpnts = oppwpnts

    def on_proj_trajectory(self, lapcount, on_trajectory, detections, to_xy):
        """Returns a Prediction for an off-trajectory half lap, otherwise None."""
        if not self.ready or on_trajectory or len(detections) == 0:
            return None
        oppwpnts = self.oppwpnts
        if not predict_return(oppwpnts, detections, self.ego_s_doublelap, self.track_length,
                              self.kernel, to_xy):
            return None
        return Prediction(lap_count=lapcount, oppwpnts=oppwpnts,
                          markers=marker_records(oppwpnts, self.max_velocity))
