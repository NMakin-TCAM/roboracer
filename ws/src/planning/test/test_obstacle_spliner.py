"""obstacle_spliner_node on a synthetic ring track (walls r = 3 and 5 m, raceline r = 4 m, counter-clockwise),
with scans ray-cast against the walls and a round obstacle. tick() runs directly; no topics."""
import math

import numpy as np
import pytest
from PIL import Image

rclpy = pytest.importorskip('rclpy')
from geometry_msgs.msg import PoseStamped  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from sensor_msgs.msg import LaserScan  # noqa: E402

from planning.obstacle_spliner_node import ObstacleSpliner  # noqa: E402

R_IN, R_OUT, R_LINE = 3.0, 5.0, 4.0
RES, HALF = 0.05, 6.0   # map: 12 x 12 m around the origin
HEADER = 's_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2'
HALF_WIDTH = 0.17   # the node's vehicle_half_width default
POCKET = (5.3, 5.8)


@pytest.fixture(scope='module')
def track(tmp_path_factory):
    d = tmp_path_factory.mktemp('ring')
    n = int(2 * HALF / RES)
    c = (np.arange(n) + 0.5) * RES - HALF
    x, y = np.meshgrid(c, c[::-1])   # image row 0 is the top
    r = np.hypot(x, y)
    img = np.full((n, n), 205, np.uint8)   # unknown, as map_saver writes it
    img[(r > R_IN) & (r < R_OUT)] = 254
    img[(r > POCKET[0]) & (r < POCKET[1])] = 254   # free space the lidar saw beyond the outer wall
    img[((r > R_IN - 0.1) & (r <= R_IN)) | ((r >= R_OUT) & (r < R_OUT + 0.1))] = 0
    Image.fromarray(img).save(d / 'map.pgm')
    (d / 'map.yaml').write_text(f'image: map.pgm\nmode: trinary\nresolution: {RES}\norigin: [{-HALF}, {-HALF}, 0.0]\n'
                                'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')
    t = np.linspace(0, 2 * np.pi, 252)   # the last row repeats the first, as TUM writes it
    rows = np.column_stack([R_LINE * t, R_LINE * np.cos(t), R_LINE * np.sin(t), t, np.full_like(t, 0.25),
                            np.full_like(t, 3.0), np.zeros_like(t)])
    np.savetxt(d / 'raceline.csv', rows, delimiter='; ', fmt='%.7f', header=HEADER)
    return d


@pytest.fixture(scope='module')
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node(ros, track):
    n = ObstacleSpliner(parameter_overrides=[Parameter('raceline_path', value=str(track / 'raceline.csv')),
                                             Parameter('map_yaml', value=str(track / 'map.yaml'))])
    n.lidar = (0.0, 0.0, 0.0)   # no TF here; test_lidar_mount_offset sets its own
    n.published = []
    n.publish = lambda xy, valid: n.published.append((np.array(xy), valid))
    yield n
    n.destroy_node()


def on_ring(angle, r=R_LINE):
    return np.array([r * math.cos(angle), r * math.sin(angle)])


def ray_circle(p, u, c, radius):
    """Smallest t > 0 with |p + t u - c| = radius, or inf."""
    b = np.dot(u, p - c)
    disc = b * b - (np.dot(p - c, p - c) - radius ** 2)
    if disc < 0:
        return math.inf
    roots = [t for t in (-b - math.sqrt(disc), -b + math.sqrt(disc)) if t > 1e-6]
    return min(roots, default=math.inf)


def tick(node, car_angle, obstacle=None, frame='map', car_r=R_LINE):
    """Car at car_angle, car_r from the origin, heading along the ring; obstacle: (angle, r, radius).
    The scan is cast from the lidar, node.lidar[0] m ahead of the car."""
    pos, yaw = on_ring(car_angle, car_r), car_angle + math.pi / 2
    pose = PoseStamped()
    pose.header.frame_id = frame
    pose.pose.position.x, pose.pose.position.y = float(pos[0]), float(pos[1])
    pose.pose.orientation.z, pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    scan = LaserScan()
    scan.angle_min, scan.angle_max, scan.angle_increment = -2.35, 2.35, math.radians(0.25)
    scan.range_min, scan.range_max = 0.05, 30.0
    circles = [(np.zeros(2), R_IN), (np.zeros(2), R_OUT)]
    if obstacle:
        circles.append((on_ring(obstacle[0], obstacle[1]), obstacle[2]))
    angles = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment)
    lidar = pos + node.lidar[0] * np.array([math.cos(yaw), math.sin(yaw)])
    scan.ranges = [float(min(ray_circle(lidar, np.array([math.cos(yaw + a), math.sin(yaw + a)]), c, rad)
                             for c, rad in circles)) for a in angles]
    node.pose_callback(pose)
    node.scan_callback(scan)
    node.tick()
    return node.published[-1]


def radii(xy):
    return np.hypot(xy[:, 0], xy[:, 1])


def test_clear_track_is_the_raceline(node):
    xy, valid = tick(node, 0.0)
    assert valid
    assert node.scan_clusters() == []   # the walls are in the map: not obstacles
    assert np.allclose(xy, node.base_xy)


def test_free_space_beyond_the_walls_is_off_the_track(node):
    assert node.clearances(on_ring(0.3, R_LINE))[0] > 0.9
    assert node.clearances(on_ring(0.3, sum(POCKET) / 2))[0] == 0.0


def test_pose_in_another_frame_is_ignored(node):
    _, valid = tick(node, 0.0, frame='world')
    assert not valid


def test_passes_an_obstacle_on_the_line(node):
    obstacle = (2.0 / R_LINE, R_LINE, 0.15)   # 2 m ahead along the line, 0.3 m across
    xy, valid = tick(node, 0.0, obstacle)
    assert valid
    centre = on_ring(obstacle[0])
    # the car's centre line keeps clear of it by at least half the car's width, all along
    assert np.min(np.linalg.norm(xy - centre, axis=1)) > obstacle[2] + HALF_WIDTH
    # beside it the path is shifted for the obstacle's length and more, not only at its centre
    beside = np.abs(np.arctan2(xy[:, 1], xy[:, 0]) - obstacle[0]) * R_LINE < 0.4
    assert np.all(np.abs(radii(xy[beside]) - R_LINE) > 0.4)
    # inside the walls, and the raceline elsewhere
    assert np.all((radii(xy) > R_IN + HALF_WIDTH) & (radii(xy) < R_OUT - HALF_WIDTH))
    far = np.abs(np.angle(np.exp(1j * (np.arctan2(xy[:, 1], xy[:, 0]) - obstacle[0])))) > math.pi / 2
    assert np.allclose(xy[far], node.base_xy[far])


def test_keeps_the_path_while_beside_the_obstacle(node):
    obstacle = (2.0 / R_LINE, R_LINE, 0.15)
    approach, _ = tick(node, 0.0, obstacle)
    side = np.sign(radii(approach)[np.argmax(np.abs(radii(approach) - R_LINE))] - R_LINE)
    car_angle = obstacle[0] + 0.2 / R_LINE   # on the shifted path, 0.2 m past the obstacle's centre
    xy, valid = tick(node, car_angle, obstacle, car_r=R_LINE + side * 0.45)
    assert valid
    near_car = np.abs(np.arctan2(xy[:, 1], xy[:, 0]) - car_angle) * R_LINE < 0.2
    assert np.all(side * (radii(xy[near_car]) - R_LINE) > 0.3)   # still shifted, to the same side


def test_obstacle_next_to_a_wall_is_seen(node):
    # 0.13 m from the outer wall: its returns run on into the wall's, and must not be taken for the wall
    obstacle = (2.0 / R_LINE, R_OUT - 0.28, 0.15)
    for car_angle in np.arange(0.0, obstacle[0] - 0.1, 0.05):   # every tick of the approach
        _, valid = tick(node, car_angle, obstacle)
        assert valid
        seen = node.scan_clusters()
        assert len(seen) == 1
        assert np.max(np.linalg.norm(seen[0] - on_ring(obstacle[0], obstacle[1]), axis=1)) < obstacle[2] + 0.02


def test_long_cluster_beside_the_line_does_not_block(node, monkeypatch):
    # 1.5 m of wall seen 0.6 m left of the raceline where the map has it farther out (a stretched map):
    # its points are out of the car's way, though a circle around them would reach across the line
    t = np.linspace(1.5, 3.0, 60) / R_LINE
    wall = np.column_stack(((R_LINE + 0.6) * np.cos(t), (R_LINE + 0.6) * np.sin(t)))
    tick(node, 0.0)
    monkeypatch.setattr(node, 'scan_clusters', lambda: [wall])
    node.tick()
    xy, valid = node.published[-1]
    assert valid
    assert np.allclose(xy, node.base_xy)


def test_far_returns_need_more_clearance_from_the_walls(node):
    # within 0.24 m of the outer wall: a map that far off would put a wall there, so it counts only up close
    node.heading_tolerance = math.radians(2.0)
    obstacle = (1.0, R_OUT - 0.14, 0.1)
    tick(node, obstacle[0] - 4.0 / R_LINE, obstacle)    # ~4 m away: 0.12 + 4 tan(2 deg) = 0.26 m needed
    assert node.scan_clusters() == []
    tick(node, obstacle[0] - 1.0 / R_LINE, obstacle)    # ~1 m away: 0.16 m needed
    assert len(node.scan_clusters()) == 1


def test_lidar_mount_offset(node):
    # the RoboRacer's lidar is 0.27 m ahead of base_link: returns land where they hit, not 0.27 m short
    node.lidar = (0.27, 0.0, 0.0)
    obstacle = (2.0 / R_LINE, R_LINE, 0.15)
    tick(node, 0.0, obstacle)
    seen = node.scan_clusters()
    assert len(seen) == 1
    assert np.all(np.abs(np.linalg.norm(seen[0] - on_ring(obstacle[0]), axis=1) - obstacle[2]) < 0.02)


def test_blocked_track_is_not_valid(node):
    _, valid = tick(node, 0.0, (2.0 / R_LINE, R_LINE, 0.8))   # leaves 0.2 m each side
    assert not valid
