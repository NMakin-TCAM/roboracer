"""Differential tests: the shared ports against upstream's own, unmodified code.

The four shared files are byte-identical between ForzaETH's Predictive Spliner
and ZJU's FSDP. These tests load upstream's file under stub ROS modules, feed it
and the port the same random scenarios, and require identical output.

They need upstream's source, which is not in this repository. Point
FORZAETH_SPLINER_SRC at a directory holding the upstream files, for example a
checkout of github.com/ForzaETH/predictive-spliner's src/ or FSDP's src/gp/.
Without it every test here skips, so a fresh clone or CI is unaffected. A file
that is present but differs from the version that was ported fails, because the
comparison would then say nothing about the port.

    FORZAETH_SPLINER_SRC=~/predictive-spliner/src \\
        python3 -m pytest src/planning/test/test_upstream_differential.py -q

DIFFERENTIAL_SCENARIOS (default 300) and DIFFERENTIAL_SEED (default 1) set the
size of the run. The ports were accepted on 3,000 scenarios on each of seeds 1
and 2. Below 300 the branch-coverage assertions are skipped, since a small run
need not reach every branch.

Upstream's ROS modules are put in sys.modules only while its file is loaded,
and only those keys are restored afterwards, so the stubs never leak into
other tests in the same pytest process.
"""
import copy
import hashlib
import importlib.util
import math
import os
import random
import sys
import types
from collections import Counter

import numpy as np
import pytest

UPSTREAM_DIR = os.environ.get('FORZAETH_SPLINER_SRC', '')
N_SCENARIOS = int(os.environ.get('DIFFERENTIAL_SCENARIOS', '300'))
SEED = int(os.environ.get('DIFFERENTIAL_SEED', '1'))
COVERAGE_MIN_SCENARIOS = 300

PORTED_MD5 = {
    'opponent_trajectory.py': '6122505fb6ca6a0c6f62bc5cf7be03fc',
    'predictor_opponent_trajectory.py': 'abe61f939c29d991e9e3ec3d016089d5',
}


def upstream_file(name):
    """Path to upstream's file, skipping if it is absent and failing if it is another version."""
    if not UPSTREAM_DIR:
        pytest.skip('FORZAETH_SPLINER_SRC not set; upstream source is not in this repository')
    path = os.path.join(os.path.expanduser(UPSTREAM_DIR), name)
    if not os.path.isfile(path):
        pytest.skip(f'{path} not found')
    with open(path, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    if md5 != PORTED_MD5[name]:
        pytest.fail(f'{path} has md5 {md5}, the port is of {PORTED_MD5[name]}: '
                    'a different upstream version, so the comparison would be meaningless')
    return path


class StopSim(Exception):
    """Raised by the stub clock when the scenario has no more frames."""


# ------------------------------------------------------------------ stub messages

class Msg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class Stamp:
    def __init__(self, t):
        self.t = t

    def to_sec(self):
        return self.t


class Obstacle(Msg):
    def __init__(self, **kw):
        defaults = dict(id=0, s_start=0.0, s_end=0.0, d_right=0.0, d_left=0.0,
                        is_actually_a_gap=False, s_center=0.0, d_center=0.0, size=0.0,
                        vs=0.0, vd=0.0, s_var=0.0, d_var=0.0, vs_var=0.0, vd_var=0.0,
                        is_static=False, is_visible=True)
        defaults.update(kw)
        super().__init__(**defaults)


class ProjOppTraj(Msg):
    def __init__(self):
        self.lapcount = 0.0
        self.nrofpoints = 0.0
        self.opp_is_on_trajectory = False
        self.detections = []


class Marker:
    ADD, DELETE, DELETEALL = 0, 2, 3
    CYLINDER = 3

    def __init__(self, header=None, id=0, type=0):
        self.header = header if header is not None else Msg(frame_id='')
        self.pose = Msg(position=Msg(x=0.0, y=0.0, z=0.0),
                        orientation=Msg(x=0.0, y=0.0, z=0.0, w=0.0))
        self.scale = Msg(x=0.0, y=0.0, z=0.0)
        self.color = Msg(r=0.0, g=0.0, b=0.0, a=0.0)
        self.type = type
        self.action = 0
        self.id = id


class MarkerArray:
    def __init__(self):
        self.markers = []


def stub_modules(world, msg_types):
    """The modules upstream imports, bound to one scenario's world."""
    def module(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    f110msg = module('f110_msgs.msg', **msg_types)
    navm = module('nav_msgs.msg', Odometry=Msg)
    stdm = module('std_msgs.msg', String=Msg)
    vism = module('visualization_msgs.msg', Marker=Marker, MarkerArray=MarkerArray)

    class FrenetConverter:
        """Identity: markers then carry (s, d), which is what the ports hand the node."""

        def __init__(self, x, y):
            pass

        def get_cartesian(self, s, d):
            return np.array([s, d])

    fcc = module('frenet_converter.frenet_converter', FrenetConverter=FrenetConverter)

    class Time:
        @staticmethod
        def now():
            return Stamp(world.now)

    class Publisher:
        def __init__(self, topic, typ, queue_size=None):
            self.topic = topic

        def publish(self, msg):
            world.log.append((world.frame, self.topic, copy.deepcopy(msg)))

    def subscriber(topic, typ, cb):
        world.subs.setdefault(topic, []).append(cb)

    class Rate:
        def __init__(self, r):
            pass

        def sleep(self):
            pass

    rospy = module('rospy', init_node=lambda *a, **k: None, Time=Time, Publisher=Publisher,
                   Subscriber=subscriber, get_param=lambda name, default=None: default,
                   wait_for_message=world.wait_for_message, is_shutdown=world.next_frame_main,
                   Rate=Rate, loginfo=lambda *a: None, spin=lambda: None,
                   Header=lambda **kw: Msg(**kw))
    return {
        'rospy': rospy,
        'f110_msgs': module('f110_msgs', msg=f110msg), 'f110_msgs.msg': f110msg,
        'nav_msgs': module('nav_msgs', msg=navm), 'nav_msgs.msg': navm,
        'std_msgs': module('std_msgs', msg=stdm), 'std_msgs.msg': stdm,
        'visualization_msgs': module('visualization_msgs', msg=vism),
        'visualization_msgs.msg': vism,
        'frenet_converter': module('frenet_converter', frenet_converter=fcc),
        'frenet_converter.frenet_converter': fcc,
    }


def load_upstream(path, stubs, name):
    """Execute upstream's file with the stubs visible, then put back only those keys.

    Not mock.patch.dict(sys.modules): that restores the whole dict, evicting the
    real modules (sklearn, scipy, matplotlib) upstream imports for the first time,
    so every scenario re-imported them. That leaked memory and could crash in C
    extensions.
    """
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


def same(a, b):
    """Exact equality over nested tuples, with NaN equal to NaN."""
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


# ================================================================== opponent_trajectory

OT_TYPES = dict(ObstacleArray=Msg, OpponentTrajectory=Msg, Obstacle=Obstacle, OppWpnt=Msg,
                WpntArray=Msg, ProjOppTraj=ProjOppTraj, ProjOppPoint=Msg)
POINT_FIELDS = ('s', 'd', 'vs', 'vd', 'is_static', 'is_visible', 'time', 's_var', 'd_var',
                'vs_var', 'vd_var')


def ot_scenario(rng):
    """A track, and frames of opponent, ego and message events with dropouts and noise."""
    L = rng.uniform(15.0, 40.0)
    spacing = rng.choice([0.1, 0.2, 0.3, 0.5])
    n = int(L / spacing)
    s_wp = [i * L / n for i in range(n + 1)]        # last duplicates the first, at s = L
    if rng.random() < 0.2:
        s_wp = s_wp[:-1]                             # sometimes no duplicate
    w = rng.uniform(0.6, 1.6)
    wpnts = [Msg(s_m=s, x_m=s, y_m=0.0, d_left=w + rng.uniform(-0.1, 0.1),
                 d_right=w + rng.uniform(-0.1, 0.1)) for s in s_wp]

    frames = []
    t = t0 = rng.uniform(0.0, 100.0)
    opp_s = rng.uniform(0, L)
    gap = rng.uniform(0.5, 5.0)
    v = rng.uniform(1.0, 9.0)
    amp, k = rng.uniform(0, 0.8), rng.uniform(0.1, 1.0)
    n_frames = rng.randint(200, 1400)
    dropout = 0
    stamp = t
    p_static = rng.random() * 0.3       # a cone in the array (breaks the opponent pick)
    p_multi = rng.random() * 0.3        # a second dynamic obstacle
    p_jump = rng.random() * 0.02        # detection jumps in s
    p_pause = rng.random() * 0.01       # long cycle, for staleness
    p_dup = rng.random() * 0.05         # repeated header stamp
    p_traj = rng.random() * 0.1         # an /opponent_trajectory message
    p_state = rng.random() * 0.02       # a state machine change
    p_wp = rng.choice([0.0, 0.0, 0.002])    # waypoints republished with another length
    off_bias = rng.choice([0.0, 0.0, 0.5, 1.0])
    for _ in range(n_frames):
        dt = rng.choice([0.02, 0.04, 0.04, rng.uniform(0.005, 0.08)])
        if rng.random() < p_pause:
            dt = rng.uniform(0.3, 4.0)
        t += dt
        v = min(max(v + rng.gauss(0, 0.2), 0.0), 16.0)
        opp_s += v * dt
        if rng.random() < p_jump:
            opp_s += rng.uniform(-3.0, 5.0)
        gap = min(max(gap + rng.gauss(0, 0.1), -1.0), 8.0)
        ego_s = opp_s - gap
        opp_d = amp * math.sin(k * opp_s) + (rng.uniform(-2, 2) if rng.random() < 0.01 else 0)
        f = {'t': t}
        f['ego'] = ((ego_s % L) if rng.random() > 0.002 else ego_s, rng.uniform(-0.5, 0.5),
                    v, 0.0)
        if dropout > 0:
            dropout -= 1
            if rng.random() < 0.5:
                f['obs'] = ([], t)              # empty array still published
        else:
            if rng.random() < 0.01:
                dropout = rng.randint(1, 60)
            if rng.random() > p_dup:
                stamp = t + rng.uniform(-0.002, 0.0)
            noise = rng.gauss(0, 0.03)
            s_c = (opp_s + noise) % L
            if rng.random() < 0.002:
                s_c = opp_s + noise             # unwrapped, past the lap
            opp = Obstacle(s_start=s_c - 0.25, s_center=s_c, d_center=opp_d,
                           vs=v + rng.gauss(0, 0.3), vd=rng.gauss(0, 0.2),
                           s_var=rng.random(), d_var=rng.random(), vs_var=rng.random(),
                           vd_var=rng.random(), is_visible=rng.random() > 0.1)
            obs = [opp]
            if rng.random() < p_static:
                obs.append(Obstacle(s_start=(ego_s + rng.uniform(0, 8)) % L, is_static=True))
            if rng.random() < p_multi:
                obs.append(Obstacle(s_start=(ego_s + rng.uniform(0, 8)) % L,
                                    s_center=(ego_s + rng.uniform(0, 8)) % L))
            rng.shuffle(obs)
            f['obs'] = (obs, stamp)
        if rng.random() < p_traj:
            m = rng.randint(0, 60)
            f['traj'] = [Msg(s_m=i * L / max(m, 1), d_m=amp * math.sin(k * i * L / max(m, 1))
                             + off_bias * rng.choice([0, 1]) + rng.gauss(0, 0.1))
                         for i in range(m)]
        if rng.random() < p_wp:
            scale = rng.uniform(0.9, 1.1)
            f['wpnts'] = [Msg(s_m=w.s_m * scale, x_m=w.x_m, y_m=0.0, d_left=w.d_left,
                              d_right=w.d_right) for w in wpnts]
        if rng.random() < p_state:
            f['state'] = rng.choice(['TRAILING', 'OVERTAKE', 'GB_TRACK'])
        frames.append(f)
    return wpnts, ([], t0), frames


def obs_msg(obs, stamp):
    return Msg(header=Msg(stamp=Stamp(stamp)), obstacles=obs)


def odom(e):
    return Msg(pose=Msg(pose=Msg(position=Msg(x=e[0], y=e[1]))),
               twist=Msg(twist=Msg(linear=Msg(x=e[2], y=e[3]))))


class OTWorld:
    """Feeds one frame per upstream loop pass: rospy.is_shutdown() in the main loop,
    time.sleep() in the off-trajectory wait. The port gets one step() per frame."""

    def __init__(self, scen):
        self.wpnts, init_obs, self.frames = scen
        self.initial = {'/global_waypoints': Msg(wpnts=self.wpnts),
                        '/perception/obstacles': obs_msg(*init_obs)}
        self.subs = {}
        self.log = []
        self.frame = -1
        self.now = init_obs[1]

    def deliver(self, topic, msg):
        for cb in self.subs.get(topic, []):
            cb(msg)

    def wait_for_message(self, topic, typ):
        msg = self.initial[topic]
        self.deliver(topic, msg)
        return msg

    def _apply(self):
        self.frame += 1
        if self.frame >= len(self.frames):
            return False
        f = self.frames[self.frame]
        if 'obs' in f:
            self.deliver('/perception/obstacles', obs_msg(*f['obs']))
        self.deliver('/car_state/odom_frenet', odom(f['ego']))
        if 'traj' in f:
            self.deliver('/opponent_trajectory', Msg(oppwpnts=f['traj'], lap_count=0.0))
        if 'state' in f:
            self.deliver('/state_machine', Msg(data=f['state']))
        if 'wpnts' in f:
            self.deliver('/global_waypoints', Msg(wpnts=f['wpnts']))
        self.now = f['t']
        return True

    def next_frame_main(self):
        return not self._apply()

    def sleep(self, _):
        if not self._apply():
            raise StopSim


def ot_norm_traj(lapcount, nrofpoints, on, dets):
    return (float(lapcount), float(nrofpoints), bool(on),
            tuple(tuple(float(getattr(p, f)) for f in POINT_FIELDS) for p in dets))


def ot_run_upstream(path, scen):
    world = OTWorld(scen)
    mod = load_upstream(path, stub_modules(world, OT_TYPES), 'upstream_opponent_trajectory')
    mod.time = types.SimpleNamespace(sleep=world.sleep)
    node = mod.Opponent_Trajectory()
    crash = None
    try:
        node.make_opponent_trajectory()
    except StopSim:
        pass
    except Exception as e:      # an upstream bug: only the frames before it compare
        crash = (world.frame, type(e).__name__)
    trajs, marks = [], []
    for frame, topic, msg in world.log:
        if topic == '/proj_opponent_trajectory':
            trajs.append((frame, ot_norm_traj(msg.lapcount, msg.nrofpoints,
                                              msg.opp_is_on_trajectory, msg.detections)))
        else:
            marks.append((frame, tuple((m.id, float(m.pose.position.x),
                                        float(m.pose.position.y),
                                        m.action == Marker.DELETEALL) for m in msg.markers)))
    return trajs, marks, crash


def ot_instrument(tr, counts):
    """Count the branches the run goes through, so a pass also shows what it exercised."""
    orig_restart, orig_project, orig_half = tr._restart, tr.project, tr._check_half_lap

    def restart(points_in_lap):
        counts['stale_flush_published' if tr.points_in_lap > 5 else 'stale_dropped'] += 1
        return orig_restart(points_in_lap)

    def project(old_index, index):
        before = tr.first_point
        r = orig_project(old_index, index)
        if not before and r[3]:
            counts['jump_over_1m'] += 1
        if not before and old_index is not None:
            diff = tr.raw[index][0] - tr.raw[old_index][0]
            if diff < 0 and abs(diff) > tr.treshold:
                counts['wrap'] += 1
            elif diff < 0:
                counts['backwards'] += 1
        return r

    def half(out):
        n_before, init_before = len(tr.detections), tr.initial_lap
        orig_half(out)
        if init_before and tr.initial_lap and len(tr.detections) < n_before:
            counts['first_half_lap_trimmed'] += 1

    tr._restart, tr.project, tr._check_half_lap = restart, project, half


def ot_run_port(scen, counts, last_frame=None):
    from planning import opponent_trajectory as ot
    wpnts, init_obs, frames = scen
    tr = ot.OpponentTrajectoryTracker()
    ot_instrument(tr, counts)
    tr.set_global_waypoints(wpnts)
    tr.set_obstacles(*init_obs)
    trajs, marks = [], []
    for i, f in enumerate(frames):
        if last_frame is not None and i > last_frame:
            break
        if 'obs' in f:
            tr.set_obstacles(*f['obs'])
        tr.set_ego(*f['ego'])
        if 'traj' in f:
            tr.set_opponent_trajectory(f['traj'])
        if 'state' in f:
            tr.set_state(f['state'])
        if 'wpnts' in f:
            tr.set_global_waypoints(f['wpnts'])
        was_waiting = tr.waiting is not None
        out = tr.step(f['t'])
        if tr.waiting is not None and not was_waiting:
            counts['off_trajectory_wait'] += 1
        if tr.raw is not None and np.isnan(tr.raw).any():
            counts['nan_row_frames'] += 1
        for t in out.trajectories:
            if t.lapcount >= 1 and (t.lapcount * 2) % 1 != 0:
                counts['fractional_lapcount'] += 1
            counts['invisible_detections_published'] += sum(not p.is_visible
                                                            for p in t.detections)
            trajs.append((i, ot_norm_traj(t.lapcount, t.nrofpoints, t.opp_is_on_trajectory,
                                          t.detections)))
        for ms in out.markers:
            marks.append((i, tuple((m.id, float(m.s), float(m.d), m.delete_all) for m in ms)))
    return trajs, marks


def test_opponent_trajectory_matches_upstream():
    path = upstream_file('opponent_trajectory.py')
    counts, differing = Counter(), []
    for i in range(N_SCENARIOS):
        scen = ot_scenario(random.Random(SEED * 1_000_003 + i))
        up_t, up_m, crash = ot_run_upstream(path, scen)
        po_t, po_m = ot_run_port(scen, counts, None if crash is None else crash[0])
        if crash is not None:
            counts['upstream_crash_' + crash[1]] += 1
            keep = lambda xs: [x for x in xs if x[0] < crash[0]]
            up_t, up_m, po_t, po_m = keep(up_t), keep(up_m), keep(po_t), keep(po_m)
        counts['published'] += len(up_t)
        if not (same(tuple(up_t), tuple(po_t)) and same(tuple(up_m), tuple(po_m))):
            differing.append(i)
    print(f'\nopponent_trajectory: {N_SCENARIOS} scenarios, seed {SEED}: {dict(counts)}')
    assert differing == [], f'scenarios {differing[:10]} differ from upstream'

    if N_SCENARIOS >= COVERAGE_MIN_SCENARIOS:
        for branch in ('wrap', 'backwards', 'jump_over_1m', 'first_half_lap_trimmed',
                       'stale_flush_published', 'stale_dropped', 'off_trajectory_wait',
                       'nan_row_frames', 'fractional_lapcount',
                       'invisible_detections_published', 'upstream_crash_AttributeError'):
            assert counts[branch] > 0, f'the run never reached {branch}'


# ================================================================== predictor_opponent_trajectory

PRED_SCENARIOS = int(os.environ.get('DIFFERENTIAL_PREDICTOR_SCENARIOS', '100'))
START = int(os.environ.get('DIFFERENTIAL_START', '0'))
PRED_COVERAGE_MIN_SCENARIOS = 100
WPNT_FIELDS = ('s_m', 'd_m', 'x_m', 'y_m', 'proj_vs_mps', 'vd_mps', 'd_var', 'vs_var')


class OppWpnt(Msg):
    def __init__(self, **kw):
        defaults = dict(id=0, s_m=0.0, d_m=0.0, x_m=0.0, y_m=0.0, proj_vs_mps=0.0, vd_mps=0.0,
                        d_var=0.0, vs_var=0.0)
        defaults.update(kw)
        super().__init__(**defaults)


class OpponentTrajectory(Msg):
    def __init__(self):
        self.header = Msg(seq=0, stamp=None, frame_id='')
        self.id = 0
        self.lap_count = 0.0
        self.opp_is_on_trajectory = False
        self.oppwpnts = []


PRED_TYPES = dict(ObstacleArray=Msg, OpponentTrajectory=OpponentTrajectory, Obstacle=Obstacle,
                  Wpnt=Msg, OppWpnt=OppWpnt, WpntArray=Msg, ProjOppTraj=ProjOppTraj,
                  ProjOppPoint=Msg)


def pred_scenario(rng):
    """Global waypoints, then a sequence of GP trajectories and ProjOppTraj half laps."""
    L = rng.uniform(10.0, 40.0)
    n = int(L / rng.choice([0.1, 0.2, 0.3, 0.5]))
    ego_s = [i * L / n for i in range(n + 1)]       # last duplicates the first
    top = 0.0 if rng.random() < 0.01 else rng.uniform(3.0, 9.0)
    wpnts = [Msg(s_m=s, x_m=s, y_m=0.0, vx_mps=top * rng.uniform(0.5, 1.0)) for s in ego_s]
    amp, k = rng.uniform(0, 0.8), rng.uniform(0.1, 1.0)

    def gp_trajectory():
        """As the main GP lays it out: one waypoint per ego s, minus the duplicate."""
        r = rng.random()
        if r < 0.03:
            return []
        s_list = ego_s[:-1]
        if r < 0.10:
            s_list = sorted(rng.sample(s_list, max(1, len(s_list) // 3)))
        ws = [OppWpnt(s_m=s + (1e-6 if r < 0.15 and rng.random() < 0.3 else 0.0),
                      d_m=amp * math.sin(k * s) + rng.gauss(0, 0.05), x_m=rng.uniform(-9, 9),
                      y_m=rng.uniform(-9, 9), proj_vs_mps=rng.uniform(2, 8),
                      vd_mps=rng.gauss(0, 0.3), d_var=rng.random() * 0.2,
                      vs_var=rng.random()) for s in s_list]
        if rng.random() < 0.05:
            rng.shuffle(ws)                  # list order, not s order, picks the 15
        return ws

    def half_lap():
        on = rng.random() < 0.25
        m = 0 if rng.random() < 0.05 else rng.randint(1, 60)
        s = rng.uniform(0, L)
        if rng.random() < 0.3:
            s = L - rng.uniform(0, 3.0)      # crosses the start line
        off = rng.choice([-1, 1]) * rng.uniform(0.3, 1.0)
        dets = []
        for i in range(m):
            s += rng.uniform(0.05, 0.5) if rng.random() > 0.03 else -rng.uniform(0, 0.3)
            dets.append(Msg(s=s % L, d=amp * math.sin(k * s) + off * (1 - i / max(m, 1)),
                            vs=rng.uniform(2, 8), vd=rng.gauss(0, 0.5), is_static=False,
                            is_visible=True, time=float(i), s_var=0.0, d_var=0.0,
                            vs_var=0.0, vd_var=0.0))
        msg = ProjOppTraj()
        msg.lapcount = rng.choice([1.0, 1.5, 2.0, rng.uniform(1, 5)])
        msg.nrofpoints = float(m)
        msg.opp_is_on_trajectory = on
        msg.detections = dets
        return msg

    events = []
    if rng.random() < 0.1:
        events.append(('proj', half_lap()))  # before any trajectory: ignored
    events.append(('traj', gp_trajectory()))
    for _ in range(rng.randint(2, 10)):
        r = rng.random()
        if r < 0.2:
            events.append(('traj', gp_trajectory()))
        elif r < 0.23:
            scale = rng.uniform(0.95, 1.05)
            events.append(('wpnts', [Msg(s_m=w.s_m * scale, x_m=w.x_m, y_m=0.0,
                                         vx_mps=w.vx_mps) for w in wpnts]))
        else:
            events.append(('proj', half_lap()))
    return wpnts, events


class PredWorld:
    """Upstream blocks in wait_for_message; the stub hands it the next event of that topic."""

    TOPIC = {'traj': '/opponent_trajectory', 'proj': '/proj_opponent_trajectory',
             'wpnts': '/global_waypoints'}

    def __init__(self, scen):
        self.wpnts, self.events = scen
        self.subs = {}
        self.log = []
        self.frame = -1
        self.now = 0.0

    def deliver(self, topic, msg):
        for cb in self.subs.get(topic, []):
            cb(msg)

    def msg_for(self, kind, payload):
        if kind == 'traj':
            m = OpponentTrajectory()
            m.oppwpnts = copy.deepcopy(payload)
            return m
        if kind == 'wpnts':
            return Msg(wpnts=payload)
        return copy.deepcopy(payload)

    def wait_for_message(self, topic, typ):
        if topic == '/global_waypoints' and self.frame < 0:
            msg = Msg(wpnts=self.wpnts)
            self.deliver(topic, msg)
            return msg
        while True:
            self.frame += 1
            if self.frame >= len(self.events):
                raise StopSim
            kind, payload = self.events[self.frame]
            msg = self.msg_for(kind, payload)
            self.deliver(self.TOPIC[kind], msg)
            if self.TOPIC[kind] == topic:
                return msg

    def next_frame_main(self):
        return self.frame + 1 >= len(self.events)

    def publish_hook(self, topic, msg):
        if topic == '/opponent_trajectory':      # upstream hears its own output
            self.deliver(topic, copy.deepcopy(msg))


def pred_marker(m):
    return (m.id, float(m.pose.position.x), float(m.pose.position.y), float(m.pose.position.z),
            float(m.scale.x), float(m.scale.y), float(m.scale.z),
            float(m.color.r), float(m.color.g), float(m.color.b), float(m.color.a))


def pred_wpnts(ws):
    return tuple(tuple(float(getattr(w, f)) for f in WPNT_FIELDS) for w in ws)


def pred_run_upstream(path, scen):
    world = PredWorld(scen)
    stubs = stub_modules(world, PRED_TYPES)
    base = stubs['rospy'].Publisher

    class Publisher(base):
        def publish(self, msg):
            super().publish(msg)
            world.publish_hook(self.topic, msg)

    stubs['rospy'].Publisher = Publisher
    mod = load_upstream(path, stubs, 'upstream_predictor_opponent_trajectory')
    node = mod.GaussianProcessOppTraj()
    crash = None
    try:
        node.get_gp_opp_traj()
    except StopSim:
        pass
    except Exception as e:      # an upstream bug: only the events before it compare
        crash = (world.frame, type(e).__name__)
    trajs, marks = [], []
    for frame, topic, msg in world.log:
        if topic == '/opponent_trajectory':
            trajs.append((frame, float(msg.lap_count), pred_wpnts(msg.oppwpnts)))
        else:
            marks.append((frame, tuple(pred_marker(m) for m in msg.markers)))
    return trajs, marks, crash


def pred_port_marker(rec):
    """The node's Marker fields, as the harness compares them."""
    rgba = (1.0, 0.0, 0.0, 0.8) if rec.predicted else (1.0, 1.0, 0.0, 1.0)
    return (rec.id, float(rec.x), float(rec.y), float(rec.height) / 2, float(rec.width),
            float(rec.width), float(rec.height)) + rgba


def pred_run_port(scen, counts, last_frame=None):
    from planning import predictor_opponent_trajectory as pot
    wpnts, events = scen
    pr = pot.ReturnPredictor()
    pr.set_global_waypoints(wpnts)
    to_xy = lambda s, d: (np.asarray(s, float), np.asarray(d, float))
    trajs, marks = [], []
    for i, (kind, payload) in enumerate(events):
        if last_frame is not None and i > last_frame:
            break
        if kind == 'traj':
            pr.set_opponent_trajectory(copy.deepcopy(payload))
        elif kind == 'wpnts':
            pr.set_global_waypoints(payload)
        else:
            dets = copy.deepcopy(payload.detections)
            acting = pr.ready and not payload.opp_is_on_trajectory and len(dets) > 0
            if acting:
                L = pr.track_length
                if any(abs(a.s - b.s) > L / 2 for a, b in zip(dets, dets[1:])):
                    counts['unwrapped_detections'] += 1
                if any(w.vs_var == pot.PREDICTED for w in pr.oppwpnts):
                    counts['built_on_own_prediction'] += 1
            elif not pr.ready:
                counts['ignored_before_ready'] += 1
            else:
                counts['ignored_on_trajectory_or_empty'] += 1
            pred = pr.on_proj_trajectory(payload.lapcount, payload.opp_is_on_trajectory,
                                         dets, to_xy)
            if acting and pred is None:
                counts['skipped_empty_span'] += 1
            if pred is not None:
                counts['predictions'] += 1
                trajs.append((i, float(pred.lap_count), pred_wpnts(pred.oppwpnts)))
                marks.append((i, tuple(pred_port_marker(m) for m in pred.markers)))
                pr.set_opponent_trajectory(copy.deepcopy(pred.oppwpnts))   # hears itself
    return trajs, marks


def test_predictor_opponent_trajectory_matches_upstream():
    sklearn = pytest.importorskip('sklearn')
    if tuple(int(x) for x in sklearn.__version__.split('.')[:2]) < (1, 1):
        pytest.skip(f'sklearn {sklearn.__version__}: before 1.1 upstream raises in '
                    'get_cartesian on every prediction, so there is nothing to compare')
    import warnings
    path = upstream_file('predictor_opponent_trajectory.py')
    counts, differing = Counter(), []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')      # sklearn ConvergenceWarning, the same on both sides
        for i in range(START, START + PRED_SCENARIOS):
            scen = pred_scenario(random.Random(SEED * 1_000_003 + i))
            up_t, up_m, crash = pred_run_upstream(path, scen)
            po_t, po_m = pred_run_port(scen, counts, None if crash is None else crash[0])
            if crash is not None:
                counts['upstream_crash_' + crash[1]] += 1
                keep = lambda xs: [x for x in xs if x[0] < crash[0]]
                up_t, up_m, po_t, po_m = keep(up_t), keep(up_m), keep(po_t), keep(po_m)
            counts['upstream_published'] += len(up_t)
            if not (same(tuple(up_t), tuple(po_t)) and same(tuple(up_m), tuple(po_m))):
                differing.append(i)
    print(f'\npredictor: scenarios {START}..{START + PRED_SCENARIOS - 1}, seed {SEED}: '
          f'{dict(counts)}')
    assert differing == [], f'scenarios {differing[:10]} differ from upstream'

    if PRED_SCENARIOS >= PRED_COVERAGE_MIN_SCENARIOS:
        for branch in ('predictions', 'unwrapped_detections', 'built_on_own_prediction',
                       'ignored_before_ready', 'ignored_on_trajectory_or_empty'):
            assert counts[branch] > 0, f'the run never reached {branch}'
