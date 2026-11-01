# One waypoint of an opponent's estimated trajectory, carrying both map-frame
# and Frenet coordinates.
#
# Field names, types and order are copied verbatim from ForzaETH's
# f110_msgs/OppWpnt so the shared Predictive Spliner / FSDP code ports with no
# edits to the consuming code.
#
# CONVENTIONS, matching Obstacle.msg:
#
#   d > 0 is LEFT of the direction of travel, matching planning.frenet.FrenetConverter.
#
#   s_m is normally in [0, lap_length), but the opponent predictor deliberately
#   extends a trajectory over a double lap by adding lap_length, so a consumer
#   may see s_m past the lap length. Compare s values with FrenetConverter.ds().
#
#   The *_var fields are the variance of the ESTIMATE at this waypoint (the
#   Gaussian process posterior), not the physical spread of the car.

int32 id

float64 s_m
float64 d_m
float64 x_m
float64 y_m

float64 proj_vs_mps          # opponent velocity along the reference line
float64 vd_mps               # opponent velocity across it, positive to the left

float64 d_var
float64 vs_var
