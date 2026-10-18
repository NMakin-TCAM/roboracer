-- Cartographer 2D mapping for the RoboRacer (sim: autodrive_bridge, car: f1tenth_system).
-- Inputs: /scan (frame laser), /odom + TF odom -> base_link, imu (remapped to /sensors/imu/raw, frame
-- imu at base_link). Output: TF map -> odom, /tracked_pose.
-- Scan matching tuning from the first AutoDRIVE Cartographer setup (PR #3).
include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  tracking_frame = "base_link",
  published_frame = "odom",          -- odometry provides odom -> base_link
  odom_frame = "odom",
  provide_odom_frame = false,
  publish_frame_projected_to_2d = true,
  publish_tracked_pose = true,       -- /tracked_pose, base_link in map
  use_odometry = true,               -- /odom
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 1,               -- /scan
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 0,
  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-3,
  trajectory_publish_period_sec = 30e-3,
  rangefinder_sampling_ratio = 1.,
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
}

MAP_BUILDER.use_trajectory_builder_2d = true

-- Lidar range_max is 10 m. Returns between max_range and range_max are inserted as free space up to
-- missing_data_ray_length. No-return beams arrive as inf and are dropped (no free space carved).
TRAJECTORY_BUILDER_2D.min_range = 0.15
TRAJECTORY_BUILDER_2D.max_range = 9.5
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 5.0
TRAJECTORY_BUILDER_2D.use_imu_data = true   -- gyro + gravity; the IMU frame must coincide with base_link

TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.15
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window = math.rad(25.)
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.translation_delta_cost_weight = 0.1
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.rotation_delta_cost_weight = 0.1

TRAJECTORY_BUILDER_2D.submaps.num_range_data = 30
TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.resolution = 0.05

TRAJECTORY_BUILDER_2D.motion_filter.max_time_seconds = 0.5
TRAJECTORY_BUILDER_2D.motion_filter.max_distance_meters = 0.1
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = math.rad(1.0)

TRAJECTORY_BUILDER_2D.ceres_scan_matcher.occupied_space_weight = 10.
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight = 10.
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = 40.

POSE_GRAPH.optimize_every_n_nodes = 10
POSE_GRAPH.constraint_builder.min_score = 0.65
POSE_GRAPH.constraint_builder.max_constraint_distance = 15.0
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.6
-- Loop closure search around the odometry estimate. Wheel odometry drifts well under 3 m per lap;
-- a 10 m window closed loops between parallel straights in testing.
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.linear_search_window = 3.0
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.angular_search_window = math.rad(20.)
POSE_GRAPH.optimization_problem.huber_scale = 1e1
POSE_GRAPH.constraint_builder.loop_closure_translation_weight = 1.1e5
POSE_GRAPH.constraint_builder.loop_closure_rotation_weight = 1e5

return options
