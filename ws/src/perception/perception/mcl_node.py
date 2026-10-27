#!/usr/bin/env python3
"""Monte Carlo localization on the saved map: a drop-in for Cartographer localization + localization_gate.

Started by mcl.launch.py. Same outputs as localization.launch.py:
  TF map -> odom      the filter's pose in the map composed with the inverse of odom -> base_link
  /localized_pose     PoseStamped in 'map' (base_link = rear axle), published at /odom rate, and only while
                      localized, so pure pursuit does not drive before that (as with localization_gate)
Extra: /mcl/particles (PoseArray), /map (OccupancyGrid, latched) for Foxglove.

  find:      the car finds itself with a global scan-matching search (ScanMatcher), in a worker thread (the pose
             and TF keep flowing): at the start, after a sim reset, and when lost. Hybrid: first only near the
             track's raceline, heading along it either way (~5x faster); if that fits well (hybrid_min_fit) and
             is decided, the car may go at once while the search over the whole drivable area at every heading
             runs in the background to verify it (moving the car if it is clearly elsewhere, keeping places that
             fit as well as alternatives). Otherwise, or with no raceline.csv, the whole-area search decides. No ground truth is used, so the sim behaves like the real car. If several places fit the
             scan about as well, choose_pose refines from the last known pose (or the map origin, where race.sh
             puts the car). If that cannot decide, the search retries (backing off to search_period_max); after
             force_after tries it takes its best guess (and a relaxed_min_fit floor) and keeps the other places
             as alternatives, scored against the scans as the car drives until one fits clearly better (switch)
             or validate_distance m rule them out.
  converge:  the cloud starts around the pose found; until it is tight (converged_*), scans are used even
             though the car stands still, then the car may drive. Afterwards a scan is used after
             update_min_d / update_min_a of odometry motion.
  thrown off: each update scores how well the scan fits the map at the estimate. When the fit stays below
             fit_drop x its running level for fit_bad_updates updates (wheel slip, a stalled sensor stream),
             a local search re-anchors the cloud if it finds a clearly better pose; every recheck_updates
             updates a small local search catches what the fit does not show (along a straight). The car keeps
             driving through both.
  doubt:     the fit below doubt_fit x its level for doubt_updates updates in a row (beyond the local search's
             reach, or something the map does not have): a global search in the background, re-anchoring only
             if it finds the car clearly elsewhere. A low fit alone never stops the car.
  lost:      only the cloud spreading beyond lost_std_xy withdraws the pose (the car stops) and starts a search.
  sim reset: fast_bridge's /autodrive/roboracer_1/sim_reset event: the pose is withheld at once; after
             debounce s with the car still, a global search.
  manual:    a PoseWithCovarianceStamped on /initialpose (RViz, Foxglove) restarts the cloud there.
"""
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from perception.map_origin import pose2d
from perception.mcl import (Alternatives, LikelihoodMap, ParticleFilter, ScanMatcher, choose_pose, compose,
                            inverse, pose_distance, relative, select_beams, wrap)
from perception.tracks import track_dir

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
MAX_PARTICLES_SHOWN = 200
MIN_BEAMS = 10
MAP_ORIGIN = (0.0, 0.0, 0.0)   # where race.sh puts the car: the pose mapping started from
LOG_PERIOD = 2.0               # s between re-anchor log lines
SLIP_COUNT = 5                 # re-anchors in a LOG_PERIOD: the scan keeps contradicting the odometry
FIND_LOG_PERIOD = 5.0          # s between repeats of the same "cannot find the car" warning


def yaw_quaternion(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)   # z, w


class MclNode(Node):
    def __init__(self):
        super().__init__('mcl')
        p = self.declare_parameter
        track = p('track', '').value
        self.beams = p('beams', 60).value
        self.update_min_d = p('update_min_d', 0.05).value           # m of odometry between scan updates
        self.update_min_a = p('update_min_a', 0.05).value           # rad
        self.max_jump = p('max_jump', 1.0).value                    # m: a bigger odometry step is a teleport
        self.init_std_xy = p('init_std_xy', 0.15).value
        self.init_std_yaw = p('init_std_yaw', 0.10).value
        self.conv_xy = p('converged_std_xy', 0.10).value
        self.conv_yaw = p('converged_std_yaw', 0.08).value
        self.conv_updates = p('converged_updates', 3).value
        self.lost_xy = p('lost_std_xy', 0.60).value
        self.debounce = p('debounce', 0.5).value                    # s after the last reset event
        self.still_speed = p('still_speed', 0.1).value              # m/s: the car has stopped after a reset
        self.timeout = p('timeout', 10.0).value
        self.publish_particles = p('publish_particles', True).value
        self.min_fit = p('min_fit', 0.5).value                      # a search result fitting worse is no pose
        self.ambiguity = p('ambiguity', 0.97).value                 # 2nd place fitting this x the best: a tie
        self.prior_radius = p('prior_radius', 3.0).value            # m around the last known pose: tie-break
        self.search_period = p('search_period', 0.5).value          # s between global search attempts
        self.search_period_max = p('search_period_max', 2.0).value  # s, backing off while they fail
        self.force_after = p('force_after', 4).value                # failed attempts before a best guess
        self.relaxed_min_fit = p('relaxed_min_fit', 0.3).value      # min_fit once guessing
        self.yaw_scale = p('yaw_scale', 1.0).value                  # m per rad when comparing poses to priors
        self.validate_distance = p('validate_distance', 5.0).value  # m of driving to rule out alternatives
        self.validate_margin = p('validate_margin', 0.05).value     # mean fit an alternative must win by
        hybrid = p('hybrid', True).value                            # raceline-first search, verified after
        self.hybrid_min_fit = p('hybrid_min_fit', 0.8).value        # a raceline-only result fitting worse: full
        self.hybrid_seeds = p('hybrid_seeds', 15).value
        self.fit_drop = p('fit_drop', 0.75).value                   # x the fit's level: try a local search
        self.fit_bad_updates = p('fit_bad_updates', 3).value
        self.doubt_fit = p('doubt_fit', 0.4).value                  # x the fit's level for doubt_updates
        self.doubt_updates = p('doubt_updates', 10).value           # updates in a row: a background search
        self.local_radius = p('local_radius', 0.8).value            # m
        self.local_yaw = p('local_yaw', 0.26).value                 # rad
        self.snap_gain = p('snap_gain', 0.1).value                  # fit a re-anchored pose must gain
        self.recheck_updates = p('recheck_updates', 5).value        # updates between small local searches
        self.recheck_radius = p('recheck_radius', 0.5).value        # m
        self.recheck_gain = p('recheck_gain', 0.03).value           # fit gain that counts as better
        self.recheck_min_shift = p('recheck_min_shift', 0.08).value  # m, smaller offsets: the filter's job
        self.recheck_agree = p('recheck_agree', 0.1).value          # m between two checks' offsets
        self.frame = 'map'

        self.lmap = LikelihoodMap.from_yaml(os.path.join(track_dir(track), 'map.yaml'),
                                            max_dist=p('max_dist', 2.0).value)
        self.pf = ParticleFilter(
            n=p('particles', 1000).value, seed=p('seed', -1).value, sigma_hit=p('sigma_hit', 0.15).value,
            z_hit=p('z_hit', 0.8).value, z_rand=p('z_rand', 0.2).value,
            likelihood_scale=p('likelihood_scale', 0.4).value,
            noise_x=tuple(p('noise_x', [0.10, 0.02]).value), noise_y=tuple(p('noise_y', [0.05, 0.05]).value),
            noise_yaw=tuple(p('noise_yaw', [0.05, 0.10]).value))
        self.matcher = ScanMatcher(self.lmap)
        self.hybrid = False
        raceline = os.path.join(track_dir(track), 'raceline.csv')
        if hybrid and os.path.isfile(raceline):
            line = np.loadtxt(raceline, delimiter=';', comments='#', ndmin=2)[:, 1:3]   # TUM: s; x; y; ...
            self.matcher.set_track(line, radius=p('hybrid_radius', 0.6).value,
                                   yaw_window=p('hybrid_yaw', math.radians(25.0)).value)
            self.hybrid = True

        self.odom = None              # (pose, stamp msg, speed)
        self.last_odom_pose = None    # odom pose of the last filter update
        self.map_T_odom = None
        self.sensor_pose = {}         # scan frame -> pose in base_link
        self.initialized = False
        self.localized = False
        self.good = 0
        self.pending = None           # monotonic time of the last reset signal
        self.search = 'start'         # why a global search is due, None while tracking
        self.priors = [MAP_ORIGIN]    # poses the car is likely near: tie-break for an ambiguous search
        self.last_search = 0.0
        self.search_wait = self.search_period
        self.attempts = 0             # failed global searches since the search started
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.job = None               # (future, kind, odom pose at the scan, generation, start time)
        self.generation = 0           # bumped whenever a running search's result would be stale
        self.alternatives = None      # Alternatives after a guessed pick
        self.find_log = ('', 0.0)     # last "cannot find" warning and when
        self.fit_level = None         # running level of the fit while tracking
        self.fit_bad = 0
        self.doubt = 0
        self.rechecks = 0
        self.recheck_offset = None    # offset (in the car's frame) the last recheck found, waiting to agree
        self.anchors = []             # monotonic times of recent re-anchors
        self.anchor_log_t = 0.0
        self.updates = 0
        self.warn_t = 0.0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseStamped, '/localized_pose', 10)
        self.particles_pub = self.create_publisher(PoseArray, '/mcl/particles', 1)
        map_pub = self.create_publisher(OccupancyGrid, '/map', LATCHED)
        self.publish_map(map_pub)

        # first: rclpy runs ready subscriptions in creation order, so a reset closes the relay before a
        # pose that arrived with it is relayed
        self.create_subscription(String, '/autodrive/roboracer_1/sim_reset', self.reset_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.initialpose_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.create_timer(0.05, self.tick)
        self.get_logger().info(f'{self.pf.n} particles on {self.lmap.w}x{self.lmap.h} map '
                               f'({self.lmap.res:.3f} m), {len(self.matcher.candidates)} search positions'
                               + (f', {self.matcher.track_poses} raceline-first poses' if self.hybrid else
                                  ', no raceline-first search') + ', waiting for /odom and /scan')

    # ---------------- setup ----------------
    def publish_map(self, pub):
        msg = OccupancyGrid()
        msg.header.frame_id = self.frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self.lmap.res
        msg.info.width, msg.info.height = self.lmap.w, self.lmap.h
        msg.info.origin.position.x, msg.info.origin.position.y = self.lmap.ox, self.lmap.oy
        msg.info.origin.orientation.w = 1.0
        msg.data = self.lmap.occupancy_grid().tolist()
        pub.publish(msg)

    def init_filter(self, pose, reason, keep_localized=False, quiet=False, matched=False):
        """Cloud around pose. matched: a scan-matched pose, known to a few cm, so a tighter cloud (a loose one
        never tightens along a featureless straight while the car stands, and the car would wait forever).
        keep_localized: a re-anchor while driving (always matched), the car does not stop for it."""
        k = 0.3 if keep_localized or matched else 1.0
        self.pf.init_gaussian(pose, k * self.init_std_xy, k * self.init_std_yaw)
        if not keep_localized:
            self.localized = False
            self.good = 0
        self.pending = None
        self.search = None
        self.fit_bad = 0
        self.doubt = 0
        self.rechecks = 0
        self.recheck_offset = None
        self.alternatives = None
        self.generation += 1
        if not quiet:
            self.get_logger().info(f'{reason}: particles at x={pose[0]:.2f} y={pose[1]:.2f} yaw={pose[2]:.2f}')
        self.initialized = True
        self.last_odom_pose = self.odom[0]
        self.map_T_odom = compose(pose, inverse(self.odom[0]))

    def start_search(self, reason):
        """Withdraw the pose and find the car again, preferring places near where it was."""
        if self.initialized:
            self.priors = [self.pf.estimate(), MAP_ORIGIN]
        self.localized = False
        self.good = 0
        self.search = reason
        self.last_search = 0.0
        self.search_wait = self.search_period
        self.attempts = 0
        self.alternatives = None
        self.generation += 1

    # ---------------- odometry: TF and pose at the odom rate ----------------
    def odom_cb(self, msg):
        pose = pose2d(msg.pose.pose)
        v = msg.twist.twist.linear
        self.odom = (pose, msg.header.stamp, math.hypot(v.x, v.y))
        if self.map_T_odom is None:
            return
        self.broadcast(msg.header.stamp)
        if self.localized:
            x, y, yaw = compose(self.map_T_odom, pose)
            out = PoseStamped()
            out.header.stamp, out.header.frame_id = msg.header.stamp, self.frame
            out.pose.position.x, out.pose.position.y = x, y
            out.pose.orientation.z, out.pose.orientation.w = yaw_quaternion(yaw)
            self.pose_pub.publish(out)

    def broadcast(self, stamp):
        x, y, yaw = self.map_T_odom
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, self.frame, 'odom'
        t.transform.translation.x, t.transform.translation.y = x, y
        t.transform.rotation.z, t.transform.rotation.w = yaw_quaternion(yaw)
        self.tf_broadcaster.sendTransform(t)

    # ---------------- scans: the filter update ----------------
    def get_sensor_pose(self, frame):
        if frame not in self.sensor_pose:
            try:
                t = self.tf_buffer.lookup_transform('base_link', frame, Time())
            except TransformException as e:
                if time.monotonic() - self.warn_t > 2.0:
                    self.warn_t = time.monotonic()
                    self.get_logger().warning(f'waiting for TF base_link -> {frame}: {e}')
                return None
            q = t.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self.sensor_pose[frame] = (t.transform.translation.x, t.transform.translation.y, yaw)
        return self.sensor_pose[frame]

    def scan_cb(self, msg):
        if self.odom is None or self.pending is not None:
            return
        sensor = self.get_sensor_pose(msg.header.frame_id)
        if sensor is None:
            return
        ranges, angles = select_beams(msg.ranges, msg.angle_min, msg.angle_increment, self.beams,
                                      max(msg.range_min, 0.05), msg.range_max)
        if len(ranges) < MIN_BEAMS:
            return
        self.poll_search()
        if self.search is not None:
            now = time.monotonic()
            if self.job is None and now - self.last_search >= self.search_wait:
                self.last_search = now
                force = self.attempts >= self.force_after
                self.submit_search('find', ranges, angles, sensor, self.priors, force,
                                   self.relaxed_min_fit if force else self.min_fit)
            return
        odom_pose = self.odom[0]
        delta = relative(self.last_odom_pose, odom_pose)
        trans = math.hypot(delta[0], delta[1])
        if trans > self.max_jump:   # the odometry frame jumped (restart, respawn): keep the pose, then check it
            self.get_logger().warning(f'odometry jumped {trans:.2f} m (reset or respawn?): keeping the pose, '
                                      'checking it against the map')
            self.last_odom_pose = odom_pose
            self.map_T_odom = compose(self.pf.estimate(), inverse(odom_pose))
            if self.localized and self.job is None:
                self.submit_search('check', ranges, angles, sensor, [self.pf.estimate()], False, self.min_fit)
            return
        moved = trans >= self.update_min_d or abs(delta[2]) >= self.update_min_a
        if not moved and self.localized:
            return
        if moved:   # before localizing, scans are used while standing still: the start pose is known
            self.pf.predict(delta)
            self.last_odom_pose = odom_pose
        self.pf.update_sensor(ranges, angles, sensor, self.lmap)
        self.pf.resample()
        self.map_T_odom = compose(self.pf.estimate(), inverse(odom_pose))
        self.check_localized(*self.pf.spread())
        if self.localized and self.search is None:
            self.check_fit(ranges, angles, sensor, odom_pose)
        if self.localized and self.alternatives is not None and self.search is None:
            self.validate(ranges, angles, sensor, odom_pose)
        self.updates += 1
        if self.publish_particles and self.updates % 5 == 0:
            self.publish_cloud(msg.header.stamp)

    # ---------------- global search, in the worker thread ----------------
    def submit_search(self, kind, ranges, angles, sensor, priors, force, floor, odom_then=None):
        """kind 'find': the car is not localized; 'check': localized, but the scan has not fit for a while;
        'verify': the whole-area search behind a raceline-first answer. odom_then: the odometry pose at the
        scan (default: now)."""
        hybrid = self.hybrid and kind == 'find'
        self.job = (self.pool.submit(self.search_job, ranges, angles, sensor, list(priors), force, floor, hybrid),
                    kind, odom_then or self.odom[0], self.generation, time.monotonic(),
                    (ranges, angles, sensor, list(priors)))

    def search_job(self, ranges, angles, sensor, priors, force, floor, hybrid):
        """Worker thread: search_global (raceline first if hybrid), then choose_pose. Returns (stage, hypotheses,
        choice). Uses no node state but the (read-only) matcher."""
        def refine(q):
            return self.matcher.search_local(q, ranges, angles, sensor, radius=1.0, yaw_range=math.radians(10.0))

        def choose(hyps, force):
            return choose_pose(hyps, priors, self.ambiguity, self.prior_radius, self.yaw_scale, force, refine)

        if hybrid:
            hyps = self.matcher.search_global(ranges, angles, sensor, seeds=self.hybrid_seeds, track_only=True)
            if hyps and hyps[0][1] >= max(floor, self.hybrid_min_fit):
                choice = choose(hyps, False)
                if choice is not None:
                    return 'raceline', hyps, choice
        hyps = self.matcher.search_global(ranges, angles, sensor)
        if not hyps or hyps[0][1] < floor:
            return 'full', hyps, None
        return 'full', hyps, choose(hyps, force)

    def poll_search(self):
        if self.job is None or not self.job[0].done():
            return
        future, kind, odom_then, generation, t0, scan = self.job
        self.job = None
        if generation != self.generation:
            return   # a reset, /initialpose or new search since it started
        try:
            stage, hyps, choice = future.result()
        except Exception as e:   # never let a search failure kill the node: it is retried
            self.get_logger().error(f'global search failed: {e!r}')
            return
        ms = (time.monotonic() - t0) * 1000.0
        moved = relative(odom_then, self.odom[0])   # the car may have moved during the search
        if kind == 'check':
            self.on_check(choice, moved, ms)
        elif kind == 'verify':
            self.on_verify(hyps, choice, moved, ms)
        else:
            self.on_find(hyps, choice, moved, ms, stage)
            if stage == 'raceline' and self.search is None:   # found: verify with the whole-area search
                self.submit_search('verify', *scan, False, self.min_fit, odom_then=odom_then)

    def on_find(self, hyps, choice, moved, ms, stage):
        if choice is None:
            self.attempts += 1
            self.search_wait = min(self.search_wait * 1.5, self.search_period_max)
            best = ', '.join(f'({h[0][0]:.1f}, {h[0][1]:.1f}, {math.degrees(h[0][2]):.0f} deg) fit {h[1]:.2f}'
                             for h in hyps[:3])
            why = ('no pose fits the scan' if not hyps or hyps[0][1] < self.min_fit else
                   f'{sum(h[1] >= self.ambiguity * hyps[0][1] for h in hyps)} places fit about as well and the '
                   'last known pose cannot tell')
            self.warn_find(f'{self.search}: {why} (best: {best or "none"}); retrying, a best guess after '
                           f'{self.force_after} tries (or /initialpose)')
            return
        pose, alternatives, how = choice
        self.init_filter(compose(pose, moved), f'{self.search}: found the car in {ms:.0f} ms ({how}'
                         + (', near the raceline; verifying' if stage == 'raceline' else '') + ')', matched=True)
        if alternatives:
            self.alternatives = Alternatives([compose(a, moved) for a in alternatives], self.odom[0],
                                             margin=self.validate_margin, distance=self.validate_distance)

    def on_verify(self, hyps, choice, moved, ms):
        """The whole-area search behind a raceline-first answer: confirm it, move the car if it is clearly
        elsewhere, or keep the places that fit as well as alternatives for driving to settle."""
        est = self.pf.estimate()
        if choice is not None and pose_distance(compose(choice[0], moved), est, self.yaw_scale) > self.local_radius:
            pose = compose(choice[0], moved)
            self.get_logger().warning(f'verify: the whole-area search puts the car '
                                      f'{math.hypot(pose[0] - est[0], pose[1] - est[1]):.2f} m away ({choice[2]}, '
                                      f'{ms:.0f} ms): moving it there')
            self.init_filter(pose, 'verify', keep_localized=self.localized, quiet=True, matched=True)
            others = [est] + [compose(a, moved) for a in choice[1]]
        else:
            tied = [h[0] for h in hyps if hyps and h[1] >= self.ambiguity * hyps[0][1]]
            others = [q for q in (compose(t, moved) for t in tied) if pose_distance(q, est, self.yaw_scale) > 1.0]
            if others:
                self.get_logger().warning(f'verify: {len(others)} other place(s) fit as well ({ms:.0f} ms): '
                                          'checking them while driving')
            else:
                self.get_logger().info(f'verify: confirmed by the whole-area search ({ms:.0f} ms)')
        if others and self.alternatives is None:
            self.alternatives = Alternatives(others, self.odom[0], margin=self.validate_margin,
                                             distance=self.validate_distance)

    def on_check(self, choice, moved, ms):
        if choice is None or not self.localized:
            return
        pose = compose(choice[0], moved)
        est = self.pf.estimate()
        if pose_distance(pose, est, self.yaw_scale) < self.local_radius:
            return   # the car is where the filter has it: something the map does not have
        self.get_logger().warning(f'relocated {math.hypot(pose[0] - est[0], pose[1] - est[1]):.2f} m by a global '
                                  f'search ({ms:.0f} ms, {choice[2]})')
        self.init_filter(pose, 'relocate', keep_localized=True, quiet=True)

    def warn_find(self, text):
        now = time.monotonic()
        if text != self.find_log[0] or now - self.find_log[1] > FIND_LOG_PERIOD:
            self.find_log = (text, now)
            self.get_logger().warning(text)

    def validate(self, ranges, angles, sensor, odom_pose):
        """After a guessed pick: switch to an alternative that keeps fitting clearly better."""
        alt = self.alternatives
        est = self.pf.estimate()
        switch = alt.update(self.matcher, est, odom_pose, ranges, angles, sensor)
        if switch is not None:
            pose, fit_alt, fit_pick = switch
            moved = relative(alt.odom0, odom_pose)
            others = [q for q in (compose(a, moved) for a in alt.poses)
                      if pose_distance(q, pose, self.yaw_scale) > 1.0] + [est]
            self.get_logger().warning(f'the guess was wrong: switching {math.hypot(pose[0] - est[0], pose[1] - est[1]):.2f} m '
                                      f'to another place that fits (mean fit {fit_alt:.2f} vs {fit_pick:.2f})')
            self.init_filter(pose, 'switch', keep_localized=True, quiet=True)
            self.alternatives = Alternatives(others, odom_pose, margin=self.validate_margin,
                                             distance=self.validate_distance)
        elif alt.done:
            self.get_logger().info(f'guess confirmed: {alt.travelled:.1f} m of driving ruled out '
                                   f'{len(alt.poses)} other place(s)')
            self.alternatives = None

    def check_fit(self, ranges, angles, sensor, odom_pose):
        """Re-anchor the cloud when the scan stops fitting where it is (see the module docstring)."""
        est = self.pf.estimate()
        fit = self.matcher.fit(est, ranges, angles, sensor)
        if self.fit_level is None:
            self.fit_level = fit
        self.doubt = self.doubt + 1 if fit < self.doubt_fit * self.fit_level else 0
        if self.doubt >= self.doubt_updates and self.job is None:
            self.doubt = 0
            self.submit_search('check', ranges, angles, sensor, [est], False, self.min_fit)
        if fit >= self.fit_drop * self.fit_level:
            self.fit_bad = 0
            self.fit_level += 0.02 * (fit - self.fit_level)
            self.recheck(est, fit, ranges, angles, sensor, odom_pose)
            return
        self.fit_bad += 1
        if self.fit_bad < self.fit_bad_updates:
            return
        self.fit_bad = 0
        t0 = time.monotonic()
        coarse, _ = self.matcher.search_local(est, ranges, angles, sensor, radius=self.local_radius,
                                              yaw_range=self.local_yaw, step=0.1, yaw_step=math.radians(3.0))
        best, best_fit = self.matcher.search_local(coarse, ranges, angles, sensor, radius=0.1,
                                                   yaw_range=math.radians(3.0), yaw_step=math.radians(1.0))
        ms = (time.monotonic() - t0) * 1000.0
        shift = math.hypot(best[0] - est[0], best[1] - est[1])
        if best_fit >= max(fit + self.snap_gain, self.fit_drop * self.fit_level) and best_fit >= self.min_fit:
            self.reanchor(best, odom_pose, f'{shift:.2f} m, {math.degrees(wrap(best[2] - est[2])):+.1f} deg: '
                                           f'fit {fit:.2f} -> {best_fit:.2f} ({ms:.0f} ms)')
        else:   # nothing fits clearly better nearby: something the map does not have, keep tracking
            self.fit_level += 0.02 * (fit - self.fit_level)

    def recheck(self, est, fit, ranges, angles, sensor, odom_pose):
        """Every recheck_updates updates, a small local search: catches the error a low fit does not show, along
        a straight (the scan fits almost as well 0.4 m along it). Re-anchors only when two checks in a row find
        a better fit at the same offset, so noise and obstacles do not make the pose jump."""
        self.rechecks += 1
        if self.rechecks % self.recheck_updates:
            return
        best, best_fit = self.matcher.search_local(est, ranges, angles, sensor, radius=self.recheck_radius,
                                                   yaw_range=math.radians(4.0), yaw_step=math.radians(1.0))
        offset = relative(est, best)
        if best_fit < fit + self.recheck_gain or math.hypot(offset[0], offset[1]) < self.recheck_min_shift:
            self.recheck_offset = None
            return
        prev, self.recheck_offset = self.recheck_offset, offset
        if prev is None or math.hypot(offset[0] - prev[0], offset[1] - prev[1]) > self.recheck_agree:
            return
        self.recheck_offset = None
        self.reanchor(best, odom_pose, f'{math.hypot(offset[0], offset[1]):.2f} m ({offset[0]:+.2f} along, '
                                       f'{offset[1]:+.2f} across), fit {fit:.2f} -> {best_fit:.2f}')

    def reanchor(self, pose, odom_pose, what):
        """Move the cloud to a scan-matched pose without stopping the car; log at most every LOG_PERIOD s."""
        self.init_filter(pose, 're-anchor', keep_localized=True, quiet=True)
        now = time.monotonic()
        self.anchors = [t for t in self.anchors if now - t < LOG_PERIOD] + [now]
        if now - self.anchor_log_t < LOG_PERIOD:
            return
        self.anchor_log_t = now
        n = len(self.anchors)
        if n >= SLIP_COUNT:
            self.get_logger().warning(f're-anchored {n} times in {LOG_PERIOD:.0f} s, last {what}: the scan keeps '
                                      'contradicting the odometry (wheels slipping, or the car is blocked)')
        else:
            self.get_logger().warning(f're-anchored {what}')

    def check_localized(self, pos_std, yaw_std):
        if self.localized:
            if pos_std > self.lost_xy:
                self.get_logger().error(f'lost: particles spread {pos_std:.2f} m, not driving')
                self.start_search('lost')
        else:
            self.good = self.good + 1 if pos_std <= self.conv_xy and yaw_std <= self.conv_yaw else 0
            if self.good >= self.conv_updates:
                self.localized = True
                self.get_logger().info(f'localized (spread {pos_std:.3f} m, {yaw_std:.3f} rad), driving')

    def publish_cloud(self, stamp):
        msg = PoseArray()
        msg.header.stamp, msg.header.frame_id = stamp, self.frame
        step = max(1, self.pf.n // MAX_PARTICLES_SHOWN)
        for x, y, yaw in self.pf.p[::step]:
            pose = Pose()
            pose.position.x, pose.position.y = float(x), float(y)
            pose.orientation.z, pose.orientation.w = yaw_quaternion(float(yaw))
            msg.poses.append(pose)
        self.particles_pub.publish(msg)

    # ---------------- resets and manual initial poses ----------------
    def reset_cb(self, msg):
        if self.localized or self.pending is None:
            self.get_logger().warning(f'sim reset ({msg.data}): relocalizing, not driving')
        self.localized = False
        self.good = 0
        self.alternatives = None
        self.generation += 1
        self.pending = time.monotonic()

    def initialpose_cb(self, msg):
        if msg.header.frame_id not in ('', self.frame):
            self.get_logger().warning(f"/initialpose in frame '{msg.header.frame_id}', needs '{self.frame}'")
            return
        if self.odom is None:
            self.get_logger().warning('/initialpose before /odom: ignored')
            return
        self.init_filter(pose2d(msg.pose.pose), '/initialpose')

    def tick(self):
        """After a reset: search for the car once it has settled."""
        if self.pending is None or self.odom is None:
            return
        waited = time.monotonic() - self.pending
        if waited < self.debounce or (self.odom[2] >= self.still_speed and waited < self.timeout):
            return
        self.pending = None
        self.start_search('sim reset')   # the cloud was not updated since the reset: priors = the pose before it


def main():
    rclpy.init()
    node = MclNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.pool.shutdown(wait=False, cancel_futures=True)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
