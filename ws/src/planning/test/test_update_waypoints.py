"""Waypoint speed updater, checked against upstream's behaviour on a small raceline."""
import pytest

from planning.update_waypoints import GB_TRACK, WaypointUpdater

N, SPACING, PLANNED = 11, 1.0, 5.0   # s = 0..10, first and last are the same point


class FakeWpnt:
    """Stands in for f1t_msgs/Wpnt, which needs a built workspace to import."""

    def __init__(self, s_m, vx_mps):
        self.s_m = s_m
        self.vx_mps = vx_mps
        self.x_m = s_m
        self.y_m = 0.0


class FakeArray:
    def __init__(self, wpnts):
        self.wpnts = wpnts


def raceline(vx=PLANNED):
    return FakeArray([FakeWpnt(i * SPACING, vx) for i in range(N)])


@pytest.fixture
def upd():
    u = WaypointUpdater(hysteresis_time=2.0)
    u.set_scaled(raceline())
    u.set_global(raceline(vx=8.0))
    return u


def tracking(u, since=0.0):
    u.on_state(GB_TRACK, since)
    return u


def speeds(u):
    return [w.vx_mps for w in u.wpnts_msg.wpnts]


# ------------------------------------------------------------------ readiness

def test_not_ready_until_both_arrays_arrive():
    """Upstream waited for both before touching anything."""
    u = WaypointUpdater()
    assert not u.ready
    u.on_state(GB_TRACK, 0.0)
    assert u.on_odom(3.0, 7.0, now=10.0) is None
    u.set_scaled(raceline())
    assert not u.ready
    assert u.on_odom(3.0, 7.0, now=10.0) is None
    u.set_global(raceline(vx=8.0))
    assert u.ready


def test_only_the_first_scaled_array_is_used(upd):
    first = upd.wpnts_msg
    assert not upd.set_scaled(raceline(vx=1.0))
    assert upd.wpnts_msg is first


def test_only_the_first_global_array_is_used(upd):
    assert not upd.set_global(raceline(vx=20.0))
    assert upd.max_speed == 8.0


def test_global_top_speed_scales_markers(upd):
    assert upd.max_speed == 8.0
    assert upd.marker_heights() == pytest.approx([PLANNED / 8.0] * N)


def test_empty_global_array_keeps_default_top_speed():
    u = WaypointUpdater()
    assert not u.set_global(FakeArray([]))
    assert u.max_speed == 10.0


# ------------------------------------------------------------------ hysteresis

def test_nothing_updated_before_any_state_message(upd):
    """Upstream starts in GB_TRACK with no start time, so it waits for a real message."""
    assert upd.on_odom(3.0, 7.0, now=100.0) is None
    assert speeds(upd) == [PLANNED] * N


def test_nothing_updated_within_hysteresis(upd):
    tracking(upd, since=10.0)
    assert upd.on_odom(3.0, 7.0, now=11.9) is None
    assert upd.on_odom(3.0, 7.0, now=12.0) == 3


def test_repeated_gb_track_does_not_restart_the_clock(upd):
    tracking(upd, since=0.0)
    upd.on_state(GB_TRACK, 1.5)
    assert upd.on_odom(3.0, 7.0, now=2.0) == 3


def test_leaving_gb_track_stops_updates_and_resets_the_clock(upd):
    tracking(upd, since=0.0)
    upd.on_state('OVERTAKE', 5.0)
    assert upd.on_odom(3.0, 7.0, now=6.0) is None
    upd.on_state(GB_TRACK, 6.0)
    assert upd.on_odom(3.0, 7.0, now=7.0) is None
    assert upd.on_odom(3.0, 7.0, now=8.0) == 3


# ------------------------------------------------------------------ updating

def test_nearest_waypoint_gets_measured_speed(upd):
    tracking(upd)
    assert upd.on_odom(4.4, 6.5, now=3.0) == 4
    assert upd.wpnts_msg.wpnts[4].vx_mps == 6.5
    assert speeds(upd).count(PLANNED) == N - 1


def test_speed_offset_is_added():
    u = WaypointUpdater(hysteresis_time=0.0, speed_offset=0.5)
    u.set_scaled(raceline())
    u.set_global(raceline())
    u.on_state(GB_TRACK, 0.0)
    u.on_odom(2.0, 6.0, now=0.0)
    assert u.wpnts_msg.wpnts[2].vx_mps == 6.5


def test_disabled_updater_changes_nothing(upd):
    tracking(upd)
    upd.enabled = False
    assert upd.on_odom(3.0, 7.0, now=3.0) is None
    assert speeds(upd) == [PLANNED] * N


def test_endpoints_are_not_kept_in_sync(upd):
    """Upstream meant to sync the duplicated first and last waypoint and doesn't.

    Kept on purpose so both comparison branches inherit the same behaviour. If
    this test starts failing, someone changed the shared port.
    """
    tracking(upd)
    assert upd.on_odom(0.1, 7.0, now=3.0) == 0
    assert upd.wpnts_msg.wpnts[0].vx_mps == 7.0
    assert upd.wpnts_msg.wpnts[-1].vx_mps == PLANNED


def test_no_lap_wrap_when_picking_nearest(upd):
    """s just short of the lap length picks the last waypoint, not the first."""
    tracking(upd)
    assert upd.on_odom(9.8, 7.0, now=3.0) == N - 1


def test_marker_heights_follow_updates(upd):
    tracking(upd)
    upd.on_odom(5.0, 8.0, now=3.0)
    assert upd.marker_heights()[5] == pytest.approx(1.0)
