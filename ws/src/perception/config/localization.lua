-- Cartographer pure localization against a saved map (localization.launch.py loads
-- tracks/<name>/map.pbstream as a frozen trajectory). Same sensors and frames as mapping.
include "mapping.lua"

-- Keep only the most recent submaps of the localization trajectory; the frozen map is not changed.
TRAJECTORY_BUILDER.pure_localization_trimmer = {
  max_submaps_to_keep = 3,
}

-- The race starts at the mapping start pose, so the new trajectory starts at the map origin.
-- Until its first constraint to the frozen map, matches are global (whole submap), so require a
-- higher score for those.
POSE_GRAPH.optimize_every_n_nodes = 20
POSE_GRAPH.constraint_builder.sampling_ratio = 0.3
POSE_GRAPH.constraint_builder.min_score = 0.6
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.7
POSE_GRAPH.global_sampling_ratio = 0.003
POSE_GRAPH.global_constraint_search_after_n_seconds = 10.

return options
