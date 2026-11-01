# The opponent observations gathered over one lap, projected onto the global
# raceline. This is the raw input the Gaussian process fits an
# OpponentTrajectory to.
#
# Field names, types and order are copied verbatim from ForzaETH's
# f110_msgs/ProjOppTraj.
#
# lapcount and nrofpoints are float64 as in the original, even though they
# count things. lapcount is copied straight into OpponentTrajectory.lap_count,
# which is float64 too, so no conversion is needed.

float64 lapcount
float64 nrofpoints
bool opp_is_on_trajectory
ProjOppPoint[] detections
