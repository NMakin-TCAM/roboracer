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

Upstream's ROS modules are stubbed only while its file is loaded
(unittest.mock.patch.dict on sys.modules), so they never leak into other tests
in the same pytest process.
"""
import copy
import hashlib
import importlib.util
import math
import os
import random
import types
from collections import Counter
from unittest import mock

import numpy as np
import pytest

UPSTREAM_DIR = os.environ.get('FORZAETH_SPLINER_SRC', '')
N_SCENARIOS = int(os.environ.get('DIFFERENTIAL_SCENARIOS', '300'))
SEED = int(os.environ.get('DIFFERENTIAL_SEED', '1'))
COVERAGE_MIN_SCENARIOS = 300

PORTED_MD5 = {
    'opponent_trajectory.py': '6122505fb6ca6a0c6f62bc5cf7be03fc',
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
    """Execute upstream's file with the stubs visible, then take them away again."""
    with mock.patch.dict('sys.modules', stubs):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
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
