"""mcl_node on the synthetic track of test_mcl (its lap as the raceline): callbacks fed directly (no executor),
searches waited for.

Finding the car (raceline first then verified, falling back to the whole area, unique, ambiguous with a guess
checked by driving, after a sim reset), and staying localized through an odometry jump, an odometry frame jump
and a stretch the map does not have (a low fit never stops it).
"""
import math

import numpy as np
import pytest
import rclpy
from nav_msgs.msg import Odometry
from PIL import Image
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from perception.mcl import compose, inverse
from perception.mcl_node import MclNode
from test_mcl import ANGLE_INC, ANGLE_MIN, LASER, RANGE_MAX, RES, lap, make_track, pose_error, scan_from


@pytest.fixture(scope='module')
def lmap():
    return make_track()


@pytest.fixture
def tracks(tmp_path, lmap, monkeypatch):
    d = tmp_path / 'synth'
    d.mkdir()
    pix = np.full(lmap.occupied.shape, 205, dtype=np.uint8)   # unknown
    pix[lmap.free] = 254
    pix[lmap.occupied] = 0
    Image.fromarray(pix).save(d / 'map.pgm')
    (d / 'map.yaml').write_text(f'image: map.pgm\nmode: trinary\nresolution: {RES}\norigin: [0.0, 0.0, 0.0]\n'
                                'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
    rows = ['# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2']   # TUM format, as build_raceline writes it
    rows += [f'{0.08 * k:.4f}; {p[0]:.4f}; {p[1]:.4f}; 0.0; 0.0; 1.0; 0.0' for k, p in enumerate(lap())]
    (d / 'raceline.csv').write_text('\n'.join(rows) + '\n')
    monkeypatch.setenv('TRACKS_DIR', str(tmp_path))


class Car:
    """Drives the node: odometry (true pose seen through a movable odometry frame) and scans."""

    def __init__(self, lmap, **params):
        args = ['--ros-args', '-p', 'track:=synth', '-p', 'seed:=1', '-p', 'publish_particles:=false',
                '-p', 'search_period:=0.0', '-p', 'search_period_max:=0.0', '-p', 'debounce:=0.0']
        for k, v in params.items():
            args += ['-p', f'{k}:={v}']
        rclpy.init(args=args)
        self.node = MclNode()
        self.node.sensor_pose['laser'] = LASER
        self.node.tf_broadcaster.sendTransform = lambda t: None
        self.poses = []
        self.node.pose_pub.publish = self.poses.append
        self.lmap = lmap
        self.rng = np.random.default_rng(0)
        self.odom_frame = (0.0, 0.0, 0.0)   # odom pose = odom_frame * true pose
        self.stamp = 0
        self.searches = []                   # kinds of global searches the node started
        submit = self.node.submit_search

        def spy(kind, *a, **kw):
            self.searches.append(kind)
            submit(kind, *a, **kw)
        self.node.submit_search = spy

    def close(self):
        self.node.pool.shutdown(wait=True)
        self.node.destroy_node()
        rclpy.try_shutdown()

    def odom(self, true, speed=0.0):
        x, y, yaw = compose(self.odom_frame, true)
        m = Odometry()
        m.header.stamp.sec = self.stamp
        m.pose.pose.position.x, m.pose.pose.position.y = x, y
        m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        m.twist.twist.linear.x = speed
        return m

    def scan(self, true, block=None):
        m = LaserScan()
        m.header.frame_id = 'laser'
        m.angle_min, m.angle_increment = ANGLE_MIN, ANGLE_INC
        m.range_min, m.range_max = 0.1, RANGE_MAX
        r = scan_from(self.lmap, true, self.rng)
        if block is not None:   # something the map does not have, right in front: these beams end at 0.4 m
            r[block] = 0.4
        m.ranges = r.astype(np.float32).tolist()
        return m

    def step(self, true, block=None, speed=0.0):
        """One odometry message and one scan at the true pose; waits for a search the scan started."""
        self.stamp += 1
        self.node.odom_cb(self.odom(true, speed))
        self.node.scan_cb(self.scan(true, block))
        if self.node.job is not None:
            self.node.job[0].result()

    def stand(self, true, n=12):
        for _ in range(n):
            self.step(true)

    def pose(self):
        p = self.poses[-1].pose
        return p.position.x, p.position.y, 2.0 * math.atan2(p.orientation.z, p.orientation.w)


@pytest.fixture
def car(tracks, lmap):
    cars = []

    def make(**params):
        cars.append(Car(lmap, **params))
        return cars[-1]
    yield make
    for c in cars:
        c.close()


def drive(car, poses, check=None, **kw):
    for p in poses:
        car.step(p, **kw)
        if check:
            check(p)


def test_finds_the_car_at_the_start_and_drives(car):
    c = car()
    P = lap()
    c.stand(P[92])                      # beside the notch: unique; the map-origin prior is irrelevant
    assert c.node.localized and pose_error(P[92], c.pose())[0] < 0.1
    drive(c, P[93:160])
    assert pose_error(P[159], c.pose())[0] < 0.1


def test_ambiguous_start_guesses_then_driving_settles_it(car):
    """On the top straight nothing tells where along it the car is, and the map-origin prior is far away: after
    force_after failed searches the node guesses, keeps the other places, and driving toward the corner settles
    it on the true pose."""
    c = car()
    P = lap()
    start = 395
    c.stand(P[start], n=4 + 8)
    assert c.node.localized, 'never guessed'
    drive(c, P[start + 1:start + 140])   # through the corner and down the left side
    assert pose_error(P[start + 139], c.pose())[0] < 0.15


def test_odometry_jump_is_corrected_without_stopping(car):
    c = car()
    P = lap()
    c.stand(P[92])
    drive(c, P[93:110])
    # a 0.4 m jump the odometry did not see (a stalled sensor stream): from here on the odometry reads the car
    # 0.4 m further back along the track than it is (odom = J * true, J * P[110] = P[110] moved back 0.4 m)
    c.odom_frame = compose(compose(P[110], (-0.4, 0.0, 0.0)), inverse(P[110]))
    errors = []
    for p in P[110:150]:
        before = len(c.poses)
        c.step(p)
        assert c.node.localized and len(c.poses) == before + 1, 'stopped the car'
        errors.append(pose_error(p, c.pose())[0])
    assert errors[0] > 0.3 and max(errors[-20:]) < 0.1


def test_a_wrong_guess_is_switched_once_a_feature_comes_into_view(car):
    """The last known pose 1.2 m off along the featureless top straight makes the node pick the wrong place;
    the true one is kept as an alternative, and the stub on the island ahead settles it while driving."""
    c = car()
    P = lap()
    start = 395
    c.node.priors = [compose(P[start], (-1.2, 0.0, 0.0))]
    c.stand(P[start])
    assert c.node.localized and pose_error(P[start], c.pose())[0] > 0.9, 'did not pick the wrong place'
    drive(c, P[start + 1:start + 70])
    assert pose_error(P[start + 69], c.pose())[0] < 0.15


def test_a_start_with_the_view_blocked_is_found_and_corrected_once_it_clears(car):
    """Two thirds of the beams end at 0.4 m (the car stopped against something the map does not have): nothing
    reaches min_fit, so after force_after tries the relaxed floor takes a guess instead of waiting forever. The
    guess can be wrong; once the car drives on with a clear view, the pose must end up right."""
    c = car()
    P = lap()
    block = np.arange(45, 225)
    for _ in range(12):
        c.step(P[92], block=block)
    assert c.node.localized, 'still waiting'
    for p in P[93:160]:
        c.step(p)
    assert c.node.localized and pose_error(P[159], c.pose())[0] < 0.15


def test_a_teleport_the_odometry_did_not_see_is_found_by_a_background_search(car):
    """The car is picked up and put down on the far side of the track while the odometry runs on smoothly
    (nothing for the local searches to reach): a global search must find it."""
    c = car()
    P = lap()
    c.stand(P[92])
    drive(c, P[93:120])
    # from now on the true pose follows P[300:], the odometry continues from where P[119] left it
    odom_at = compose(c.odom_frame, P[119])
    c.odom_frame = compose(odom_at, inverse(P[299]))
    drive(c, P[300:360])
    assert {'check', 'find'} & set(c.searches[1:]), 'no global search'
    assert c.node.localized and pose_error(P[359], c.pose())[0] < 0.15   # on the featureless top straight


def test_a_stretch_the_map_does_not_have_never_stops_the_car(car):
    """Two thirds of the beams blocked at 0.4 m for 60 updates (a parked car, a person): the fit stays low the
    whole time and a background search checks the pose; the pose must keep flowing and stay right."""
    c = car()
    P = lap()
    c.stand(P[92])
    drive(c, P[93:100])
    block = np.arange(45, 225)
    published = []
    for p in P[100:160]:
        before = len(c.poses)
        c.step(p, block=block)
        published.append(len(c.poses) > before)
        assert c.node.localized, 'stopped the car'
    assert all(published)
    assert 'check' in c.searches
    assert pose_error(P[159], c.pose())[0] < 0.15


def test_odometry_frame_jump_keeps_the_pose(car):
    """The odometry restarts (frame jumps 3 m): the published pose must not jump with it."""
    c = car()
    P = lap()
    c.stand(P[92])
    drive(c, P[93:100])
    c.odom_frame = (3.0, -2.0, 0.5)
    drive(c, P[100:110])
    assert c.node.localized and pose_error(P[109], c.pose())[0] < 0.1


def test_sim_reset_finds_the_car_where_it_was_put(car):
    c = car()
    P = lap()
    c.stand(P[92])
    drive(c, P[93:100])
    c.node.reset_cb(String(data='collision count 0 -> 1'))
    assert not c.node.localized
    c.odom_frame = (5.0, 5.0, 1.0)      # the bridge's odometry restarts too
    c.node.odom_cb(c.odom(P[300]))
    c.node.tick()                        # debounce 0, car still: search
    c.stand(P[300])
    assert c.node.localized and pose_error(P[300], c.pose())[0] < 0.1


def test_reanchor_state_is_cleared_by_a_restart(car):
    c = car()
    P = lap()
    c.stand(P[92])
    c.node.recheck_offset = (0.3, 0.0, 0.0)
    c.node.rechecks = 3
    c.node.init_filter(P[92], 'test')
    assert c.node.recheck_offset is None and c.node.rechecks == 0


def fake_local_search(c, offsets):
    """Low-fit local searches (radius != 0.1; the 0.1 m refinement keeps its start) answer with the next offset
    from offsets, fitting perfectly: what a badly fitting part of the map can do."""
    k = [0]

    def search_local(pose, *a, radius=0.8, **kw):
        if radius == 0.1:
            return pose, 1.0
        k[0] += 1
        return compose(pose, offsets[(k[0] - 1) % len(offsets)]), 1.0
    c.node.matcher.search_local = search_local
    anchors = []
    reanchor = c.node.reanchor
    c.node.reanchor = lambda *a: (anchors.append(a[0]), reanchor(*a))
    return anchors


def test_a_low_fit_search_re_anchors_only_when_the_next_one_agrees(car):
    c = car()
    P = lap()
    c.stand(P[92])
    drive(c, P[93:100])
    block = np.arange(45, 225)   # the fit drops: every fit_bad_updates updates a local search
    anchors = fake_local_search(c, [(0.5, 0.0, 0.0), (-0.5, 0.0, 0.0)])
    drive(c, P[100:112], block=block)
    assert not anchors, 'one search moved the pose'
    anchors = fake_local_search(c, [(0.5, 0.0, 0.0)])
    drive(c, P[112:118], block=block)
    assert len(anchors) == 1


def test_a_burst_of_re_anchors_pauses_them_and_checks_in_the_background(car):
    c = car()
    P = lap()
    c.stand(P[92])
    drive(c, P[93:100])
    for _ in range(3):
        c.node.reanchor(c.node.pf.estimate(), None, 'test')
    assert c.node.burst and c.node.hold_until > 0.0
    n = len(c.searches)
    anchors = fake_local_search(c, [(0.5, 0.0, 0.0)])
    published = []
    for p in P[100:115]:
        before = len(c.poses)
        c.step(p, block=np.arange(45, 225))
        published.append(len(c.poses) > before)
    assert 'check' in c.searches[n:] and not c.node.burst
    assert not anchors, 're-anchored during the pause'
    assert all(published), 'the car waited'


def settle(c, n=6):
    """Let a background (verify) search finish and be handled."""
    for _ in range(n):
        if c.node.job is not None:
            c.node.job[0].result()
        c.node.poll_search()


def test_raceline_first_finds_the_car_and_the_whole_area_search_confirms(car):
    c = car()
    P = lap()
    c.stand(P[92])
    settle(c)
    assert c.searches[:2] == ['find', 'verify']
    assert c.node.localized and c.node.alternatives is None and pose_error(P[92], c.pose())[0] < 0.1


def test_a_car_turned_across_the_track_falls_back_to_the_whole_area(car):
    c = car()
    pose = (10.0, 1.5, math.radians(100))
    c.stand(pose)
    settle(c)
    assert 'verify' not in c.searches     # answered by the whole-area search itself
    assert c.node.localized and pose_error(pose, c.pose())[0] < 0.1 and pose_error(pose, c.pose())[1] < 0.05


def test_a_car_facing_backwards_is_never_confidently_misplaced(car):
    """On the right side facing down it sees what a car driving down the left side sees: the node must either
    place it right or keep the right place as an alternative to settle by driving."""
    c = car()
    pose = (18.55, 6.0, -math.pi / 2)
    c.stand(pose)
    settle(c)
    assert c.node.localized
    if pose_error(pose, c.pose())[0] > 0.15:
        alts = c.node.alternatives
        assert alts is not None and any(pose_error(pose, q)[0] < 0.15 for q in alts.poses)


def test_without_hybrid_the_whole_area_search_finds_the_car(car):
    c = car(hybrid='false')
    P = lap()
    c.stand(P[92])
    assert 'verify' not in c.searches and c.node.localized and pose_error(P[92], c.pose())[0] < 0.1
