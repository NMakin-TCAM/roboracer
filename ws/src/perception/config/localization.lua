-- Cartographer pure localization against a saved map (localization.launch.py loads
-- tracks/<name>/map.pbstream as a frozen trajectory). Same sensors and frames as mapping.
include "mapping.lua"

-- Keep only the most recent submaps of the localization trajectory; the frozen map is not changed.
TRAJECTORY_BUILDER.pure_localization_trimmer = {
  max_submaps_to_keep = 3,
}

-- localization_gate starts the trajectory with an initial pose relative to the frozen map, so
-- matches are local right away. Global matches (whole submap, higher score) only help when the pose
-- is wrong: on the real car or in competition a respawn is not detected and only they recover it.
POSE_GRAPH.optimize_every_n_nodes = 20
POSE_GRAPH.constraint_builder.sampling_ratio = 0.3
POSE_GRAPH.constraint_builder.min_score = 0.6
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.7
POSE_GRAPH.global_sampling_ratio = 0.003
POSE_GRAPH.global_constraint_search_after_n_seconds = 10.

-- localization_gate waits for the first optimization with a constraint to the map, which comes after
-- optimize_every_n_nodes (20) new nodes. A parked car (start, after a sim reset) adds a node only every
-- max_time_seconds: 0.5 in mapping.lua took ~10.5 s. 0.04 inserts every 2nd parked scan (republished
-- every ~27 ms): 14-20 nodes/s depending on the sim frame rate, so localized ~1.6-2.5 s after the start.
-- Below ~2 m/s (0.1 m distance trigger) moving nodes also come at up to 20/s: shorter submaps and more
-- pose-graph work than with 0.5 at low speed; faster, nothing changes.
TRAJECTORY_BUILDER_2D.motion_filter.max_time_seconds = 0.04

return options
