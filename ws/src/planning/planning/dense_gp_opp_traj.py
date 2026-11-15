"""Predictive Spliner's dense GP: the opponent's trajectory from its observed half laps.

Port of gaussian_process_opp_traj.py from ForzaETH's Predictive Spliner (the
dense sklearn GP). FSDP has a different file of the same name (a sparse GP), so
this module carries the "dense" name and only the Predictive Spliner branch
uses it.

It reads the half laps opponent_trajectory publishes on /proj_opponent_trajectory
and lays an OpponentTrajectory on the /global_waypoints s, one waypoint per
global waypoint (minus the duplicated last one), published on
/opponent_trajectory. That message is what collision_prediction rolls the
opponent forward on and what predictor_opponent_trajectory patches.

How it works, as upstream does it:

  * It starts from a whole lap of waypoints on the raceline (d = 0) with
    proj_vs_mps = 100, and fills it in place as half laps arrive. Every
    message carries the whole list, filled or not.
  * Half-lap mode (lapcount <= 1). The first half lap fits a GP in d and one in
    vs to that half lap's detections only, and predicts them on the global s
    strictly between its first detection and half a lap on. The next half lap
    predicts on the complement of that window, [end, start + lap], from its own
    detections, wherever they actually are. d_var and vs_var are the GPs'
    standard deviations, not variances.
  * Whole-lap mode (lapcount > 1, and once at lapcount == 1). d is no longer a
    GP: the accumulated detections are converted to x, y, smoothed with CCMA
    (w_ma = 5, w_cc = 3), converted back, and interpolated onto every global s.
    d_var is 0. vs is a GP over the accumulated detections, with the 3 m
    either side of the start line copied across it so the fit wraps.

The behaviour is upstream's, including its rough edges, because changing it
here would make the Predictive Spliner branch differ from the code it claims
to port:

  * Unobserved waypoints keep proj_vs_mps = 100 until the whole-lap fit. After
    the first half lap that is the half of the lap the opponent is about to
    enter.
  * lapcount == 1 exactly publishes twice. The first message is the second
    half lap's. The second is the same message object republished after the
    whole-lap fit edited its waypoints in place, so it keeps the half lap's
    lap_count (1) and header stamp. Upstream's markers for both are published.
  * At lapcount == 1 the last half lap's detections are appended to the
    accumulated list a second time (the same objects), so up to the first
    thinning they count double in the vs GP and in CCMA.
  * Detection s is edited in place. In half-lap mode, from the first decrease
    in s onwards every detection gets a lap added, and is wrapped back after
    publishing. In whole-lap mode every detection but the first is taken modulo
    the lap.
  * Thinning: once more than 200 detections are accumulated (and lapcount >= 1),
    they are binned every two waypoint spacings and only the newest of each
    bin is kept. Bins are [(x - 1) * delta, x * delta), x from 0, and stop at
    the last bin ending more than one bin below the highest s, so the
    detections in the last one to two bins before the highest are dropped every
    time it runs.
  * vs_var is misaligned in whole-lap mode. The prediction s list is the lap
    with the 3 m either side of the start line copied across it. The vs mean is
    cut back to the lap; sigma_vs is not, so each waypoint gets the sigma of a
    point about 3 m before it (at 0.1 m spacing, 30 points earlier).
  * In half-lap mode, global s past the lap is only wrapped when the detections
    crossed the start line. Otherwise those waypoints match nothing and keep
    their old values.
  * The kernels are C(0.5) * RBF(1.0) + C(0.2) * White(1.0) for vs and
    C(0.5) * Matern(1.0, nu=1.5) + C(0.2) * White(1.0) for d. Only the
    constants are bounded, to [1e-6, 1e3], each fit is one L-BFGS-B run from
    those values without restarts, and normalize_y is False, so outside the
    detections the means fall back to 0.
  * The double-lap s list, the top speed and the initial trajectory are fixed
    from the first /global_waypoints. The lap length follows the latest.
  * OppWpnt.id is never set (0 throughout), and header.seq = lap_count is
    dropped (ROS 2 has no seq).
  * Off-trajectory half laps, and half laps with no detections, are ignored.
    The predictor handles those.

Where upstream would crash, or depends on the library version, the port does
not, as in the other ports:

  * Before sklearn 1.1, GaussianProcessRegressor.predict returns the mean as
    (n, 1). Upstream's half-lap d then reaches ForzaETH's get_cartesian as a
    list of lists, and the in-place `x += d * cos(...)` raises, as in
    predictor_opponent_trajectory. The port flattens both means. From 1.1 on
    it agrees with upstream exactly.
  * If every global waypoint speed is 0, upstream's marker heights divide by
    zero. The port draws them with height 0.
  * A non-finite s, d or vs among the points about to be fitted makes
    upstream's GP fit (or, in whole-lap mode, ForzaETH's get_frenet) raise, and
    the node dies. opponent_trajectory's NaN poisoning publishes exactly that
    (vs and vd NaN) from its stale flush. In whole-lap mode, thinning may drop
    the bad point first, and then upstream carries on, as the port does. Where
    upstream would raise, the port publishes nothing for that message, removes
    the non-finite detections from the accumulated list, and wraps a half
    lap's s back into the lap as upstream would have after publishing. (At
    lapcount == 1 the half lap's own fit always fails first, so the whole-lap
    fit never fails after a publish.) A NaN vd alone does not crash upstream
    and is kept.
  * If a half lap's window holds no global s, upstream raises IndexError. The
    port publishes nothing for that half lap. Not reachable with lapcount
    0.5 then 1.0, only with other lapcounts at or below 1.

And where the port changes something on purpose:

  * The Frenet conversions (to_xy, to_sd) are FrenetConverter's, not
    ForzaETH's. In whole-lap mode d goes through x, y and back for CCMA, so
    the converter sets the published d_m, not only the markers. ForzaETH's
    get_frenet guesses s as "index of the nearest waypoint * 0.1 m" and then
    corrects it by at most 5 cm, so it is only right for 0.1 m waypoint
    spacing. Ours projects onto the polyline.

No rclpy here, so it runs under plain pytest. dense_gp_opp_traj_node.py is the
wrapper.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel

from planning.vendor.ccma import CCMA

INITIAL_VS = 100            # m/s on every waypoint before it is observed
THIN_ABOVE = 200            # detections accumulated before thinning
WRAP_MARGIN = 3.0           # m either side of the start line copied across it
CCMA_W_MA = 5
CCMA_W_CC = 3
MATCH_S = 1e-8              # m, a waypoint is written if its s is this close


class NonFiniteTraining(ValueError):
    """The points about to be fitted hold a non-finite s, d or vs; upstream raised here."""


def make_kernels():
    """Upstream's (vs, d) kernels."""
    c1_d = ConstantKernel(constant_value=0.5, constant_value_bounds=(1e-6, 1e3))
    c2_d = ConstantKernel(constant_value=0.2, constant_value_bounds=(1e-6, 1e3))
    c1_vs = ConstantKernel(constant_value=0.5, constant_value_bounds=(1e-6, 1e3))
    c2_vs = ConstantKernel(constant_value=0.2, constant_value_bounds=(1e-6, 1e3))
    kernel_vs = c1_vs * RBF(length_scale=1.0) + c2_vs * WhiteKernel(noise_level=1)
    kernel_d = c1_d * Matern(length_scale=1.0, nu=3 / 2) + c2_d * WhiteKernel(noise_level=1)
    return kernel_vs, kernel_d


def _optimizer(obj_func, initial_theta, bounds):
    solution, function_value, _ = fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds)
    return solution, function_value


@dataclass
class OppWpntRec:
    """One OppWpnt, with upstream's defaults for the fields it never sets."""

    s_m: float
    x_m: float
    y_m: float
    d_m: float = 0.0
    proj_vs_mps: float = INITIAL_VS
    vd_mps: float = 0.0
    d_var: float = 0.0
    vs_var: float = 0.0
    id: int = 0


@dataclass
class MarkerRec:
    """One /opponent_traj_markerarray cylinder (upstream's are all yellow)."""

    id: int
    x: float
    y: float
    height: float
    width: float


@dataclass
class Publication:
    """One publish on /opponent_trajectory, plus its marker array.

    oppwpnts is a snapshot taken at publish time. stamp_of says which earlier
    publication's header stamp to reuse (the lapcount == 1 republish), or None
    for a fresh stamp.
    """

    lap_count: float
    oppwpnts: list
    markers: list
    stamp_of: object = None


def marker_records(oppwpnts, max_velocity):
    out = []
    for i, w in enumerate(oppwpnts):
        height = w.proj_vs_mps / max_velocity if max_velocity else 0.0
        width = min(max(5 * w.d_var, 0.07), 0.7)
        out.append(MarkerRec(id=i, x=w.x_m, y=w.y_m, height=height, width=width))
    return out


def sort_detections(detections, sorted_list, lap_count, track_length, n_ego):
    """Upstream's create_sorted_detection_list. Returns (sorted_list, around_origin).

    Appends the detections to sorted_list (in place) and sorts by s. Edits
    detection s in place: half-lap mode adds a lap from the first decrease on,
    whole-lap mode wraps every detection but the first.
    """
    around_origin = False
    if lap_count <= 1:
        for i in range(len(detections) - 1):
            if detections[i].s > detections[i + 1].s:
                around_origin = True
            if around_origin:
                detections[i + 1].s = detections[i + 1].s + track_length
    else:
        for i in range(len(detections) - 1):
            detections[i + 1].s = detections[i + 1].s % track_length

    for det in detections:
        sorted_list.append(det)
    sorted_list.sort(key=lambda x: x.s)

    delta_s = 2 * (track_length / n_ego)
    if len(sorted_list) > THIN_ABOVE and lap_count >= 1:
        thinned = []
        last_s = sorted_list[-1].s
        i = 0
        for x in range(int(track_length / delta_s)):
            if last_s > (x + 1) * delta_s:
                helper = []
                while sorted_list[i].s < x * delta_s:
                    helper.append(sorted_list[i])
                    i = i + 1
                if len(helper) > 0:
                    helper.sort(key=lambda x: x.time)
                    thinned.append(helper[-1])
        sorted_list = thinned

    return sorted_list, around_origin


def columns(sorted_list):
    return ([det.s for det in sorted_list], [det.d for det in sorted_list],
            [det.vs for det in sorted_list], [det.vd for det in sorted_list])


def ccma_d(opp_s, opp_d, ego_s, to_xy, to_sd):
    """Whole-lap d: CCMA in x, y with one wrapped point each end, back to Frenet, onto ego_s."""
    s_copy = list(opp_s)
    d_copy = list(opp_d)
    s_copy.insert(0, s_copy[-1])
    d_copy.insert(0, d_copy[-1])
    s_copy.append(s_copy[1])
    d_copy.append(d_copy[1])

    xs, ys = to_xy(s_copy, d_copy)
    noisy = np.column_stack((np.asarray(xs, float), np.asarray(ys, float)))
    smoothed = CCMA(w_ma=CCMA_W_MA, w_cc=CCMA_W_CC).filter(noisy)

    sm_s, sm_d = to_sd(smoothed[:, 0], smoothed[:, 1])
    sm_s, sm_d = zip(*sorted(zip(sm_s, sm_d)))
    return np.interp(ego_s, sm_s, sm_d)


def wrap_training(opp_s, opp_d, opp_vs, opp_vd, ego_s_copy, track_length):
    """Upstream's copies across the start line, in place, with its loop structure.

    Prepends the detections within WRAP_MARGIN of the lap end (shifted a lap
    back) and appends those within WRAP_MARGIN of 0 (shifted a lap on), and
    does the same to the prediction s list. Returns the number of prediction s
    prepended.
    """
    n = -1
    nr_pre = 0
    for _ in range(len(opp_s)):
        if abs(opp_s[n] - track_length) < WRAP_MARGIN:
            opp_s.insert(0, opp_s[n] - track_length)
            n = n - 1
            nr_pre = nr_pre + 1
    for lst in (opp_d, opp_vs, opp_vd):
        n = -1
        for _ in range(nr_pre):
            lst.insert(0, lst[n])
            n = n - 1

    n = -1
    nr_ego = 0
    for _ in range(len(ego_s_copy)):
        if abs(ego_s_copy[n] - track_length) < WRAP_MARGIN:
            ego_s_copy.insert(0, ego_s_copy[n] - track_length)
            n = n - 1
            nr_ego = nr_ego + 1

    n = 0
    nr_app = 0
    for _ in range(len(opp_s)):
        if abs(opp_s[nr_pre + n]) < WRAP_MARGIN:
            opp_s.append(opp_s[nr_pre + n] + track_length)
            n = n + 1
            nr_app = nr_app + 1
    for lst in (opp_d, opp_vs, opp_vd):
        n = 0
        for _ in range(nr_app):
            lst.append(lst[nr_pre + n])
            n = n + 1

    n = 0
    for _ in range(len(ego_s_copy)):
        if abs(ego_s_copy[nr_ego + n]) < WRAP_MARGIN:
            ego_s_copy.append(ego_s_copy[nr_ego + n] + track_length)
            n = n + 1
    return nr_ego


def fit_and_write(whole_lap, ego_s_sorted, opp_s, opp_d, opp_vs, opp_vd, around_origin,
                  oppwpnts, track_length, kernel_vs, kernel_d, to_xy, to_sd):
    """Upstream's get_opponnent_wpnts: fit, and write the result into oppwpnts in place.

    opp_* are edited (the start-line copies) in whole-lap mode.
    """
    ego_s_copy = list(ego_s_sorted)
    if not np.isfinite([opp_s, opp_d, opp_vs]).all():
        raise NonFiniteTraining('non-finite s, d or vs in the points to fit')

    if whole_lap:
        d_pred_ccma = ccma_d(opp_s, opp_d, ego_s_sorted, to_xy, to_sd)
        wrap_training(opp_s, opp_d, opp_vs, opp_vd, ego_s_copy, track_length)

    train_s = np.array(opp_s).reshape(-1, 1)
    s_pred = np.array(ego_s_copy).reshape(-1, 1)

    gpr_vs = GaussianProcessRegressor(kernel=kernel_vs, optimizer=_optimizer)
    gpr_vs.fit(train_s, np.array(opp_vs).reshape(-1, 1))
    vs_pred, sigma_vs = gpr_vs.predict(s_pred, return_std=True)
    vs_pred = np.ravel(vs_pred)         # (n, 1) before sklearn 1.1

    if not whole_lap:
        gpr_d = GaussianProcessRegressor(kernel=kernel_d, optimizer=_optimizer)
        gpr_d.fit(train_s, np.array(opp_d).reshape(-1, 1))
        d_pred_gp, sigma_d = gpr_d.predict(s_pred, return_std=True)
        d_pred_gp = np.ravel(d_pred_gp)

    if whole_lap:
        # Cut the prediction back to the lap. sigma_vs is not cut (upstream's).
        n = 0
        for i in range(len(ego_s_copy)):
            if ego_s_copy[i - n] >= track_length or ego_s_copy[i - n] < 0:
                ego_s_copy.pop(i - n)
                vs_pred = np.delete(vs_pred, i - n)
                n += 1
        resampled_d = d_pred_ccma
    else:
        resampled_d = d_pred_gp
    resampled_vd = np.interp(ego_s_sorted, opp_s, opp_vd)

    if around_origin:
        ego_s = [ego_s_sorted[i] % track_length for i in range(len(ego_s_copy))]
    else:
        ego_s = ego_s_copy

    xs, ys = to_xy(ego_s, resampled_d.tolist())

    for w in oppwpnts:
        for j in range(len(ego_s)):
            if abs(ego_s[j] - w.s_m) < MATCH_S:
                w.x_m = xs[j]
                w.y_m = ys[j]
                w.d_m = resampled_d[j]
                w.proj_vs_mps = vs_pred[j]
                w.vd_mps = resampled_vd[j]
                w.d_var = 0 if whole_lap else sigma_d[j]
                w.vs_var = sigma_vs[j]


class DenseOpponentGP:
    """The state upstream kept on its node. on_proj_trajectory() is one loop pass.

    to_xy(s_list, d_list) -> (xs, ys) and to_sd(xs, ys) -> (s, d) are the
    Frenet conversions (see the module docstring).
    """

    def __init__(self, to_xy, to_sd):
        self.to_xy = to_xy
        self.to_sd = to_sd
        self.kernel_vs, self.kernel_d = make_kernels()
        self.track_length = None        # latest /global_waypoints
        self.ego_s_original = None      # fixed from the first, without the duplicate
        self.ego_s_doublelap = None
        self.max_velocity = None
        self.oppwpnts = None
        self.first_half_lap = True
        self.sorted_detections = []
        self.first_ego_s = None
        self.last_ego_s = None
        self.lap_count = None

    @property
    def ready(self):
        return self.oppwpnts is not None

    def set_global_waypoints(self, wpnts):
        wpnts = list(wpnts)
        if not wpnts:
            return False    # upstream would raise IndexError here
        self.track_length = wpnts[-1].s_m
        if self.ego_s_original is None:
            self.max_velocity = max(w.vx_mps for w in wpnts)
            ego_s = [w.s_m for w in wpnts]
            ego_s.pop()
            self.ego_s_original = ego_s
            self.ego_s_doublelap = ego_s + [s + self.track_length for s in ego_s]
            xs, ys = self.to_xy(ego_s, np.zeros(len(ego_s)).tolist())
            self.oppwpnts = [OppWpntRec(s_m=ego_s[i], x_m=xs[i], y_m=ys[i])
                             for i in range(len(ego_s))]
        return True

    def _publication(self, lap_count, stamp_of=None):
        return Publication(lap_count=lap_count,
                           oppwpnts=[replace(w) for w in self.oppwpnts],
                           markers=marker_records(self.oppwpnts, self.max_velocity),
                           stamp_of=stamp_of)

    def _fit(self, whole_lap, ego_s_sorted, sorted_list, around_origin):
        opp_s, opp_d, opp_vs, opp_vd = columns(sorted_list)
        fit_and_write(whole_lap, ego_s_sorted, opp_s, opp_d, opp_vs, opp_vd, around_origin,
                      self.oppwpnts, self.track_length, self.kernel_vs, self.kernel_d,
                      self.to_xy, self.to_sd)

    def _half_lap_window(self, first_s):
        L = self.track_length
        if self.first_half_lap:
            self.first_half_lap = False
            window = [s for s in self.ego_s_doublelap if first_s < s < first_s + L / 2]
        else:
            self.last_ego_s = self.last_ego_s % L
            if self.last_ego_s > self.first_ego_s:
                self.first_ego_s = self.first_ego_s + L
            window = [s for s in self.ego_s_doublelap
                      if self.last_ego_s <= s <= self.first_ego_s]
        if not window:
            return None     # upstream would raise IndexError here
        self.last_ego_s = window[-1]
        self.first_ego_s = window[0]
        return window

    def on_proj_trajectory(self, lapcount, on_trajectory, detections):
        """One /proj_opponent_trajectory message. Returns the list of Publications.

        detections are edited in place and kept, as upstream did.
        """
        if not self.ready or not (on_trajectory and len(detections) != 0):
            return []
        out = []
        try:
            self._process(lapcount, detections, out)
        except NonFiniteTraining:
            # Upstream died here. Keep what it had published, and drop the bad points.
            self.sorted_detections = [det for det in self.sorted_detections
                                      if np.isfinite([det.s, det.d, det.vs]).all()]
        return out

    def _process(self, lapcount, detections, out):
        L = self.track_length
        n_ego = len(self.ego_s_original)
        self.lap_count = lapcount

        if self.lap_count <= 1:
            this_half, around_origin = sort_detections(detections, [], self.lap_count, L, n_ego)
            for det in this_half:
                self.sorted_detections.append(det)
            window = self._half_lap_window(this_half[0].s)
            if window is None:
                return      # upstream would raise IndexError here
            try:
                self._fit(False, window, this_half, around_origin)
            except NonFiniteTraining:
                self._wrap_accumulated(around_origin)     # as upstream would have
                raise
            out.append(self._publication(self.lap_count))
            self._wrap_accumulated(around_origin)

            if self.lap_count == 1:
                published_lap_count = self.lap_count
                self.lap_count = 1.1
                self.sorted_detections, _ = sort_detections(
                    detections, self.sorted_detections, self.lap_count, L, n_ego)
                self._fit(True, self.ego_s_original, self.sorted_detections, False)
                out.append(self._publication(published_lap_count, stamp_of=len(out) - 1))
        else:
            self.sorted_detections, _ = sort_detections(
                detections, self.sorted_detections, self.lap_count, L, n_ego)
            self._fit(True, self.ego_s_original, self.sorted_detections, False)
            out.append(self._publication(self.lap_count))

    def _wrap_accumulated(self, around_origin):
        """After a half lap that crossed the start line, wrap every s back and re-sort."""
        if around_origin:
            for det in self.sorted_detections:
                det.s = det.s % self.track_length
            self.sorted_detections.sort(key=lambda x: x.s)
