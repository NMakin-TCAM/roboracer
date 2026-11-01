# An opponent's estimated trajectory over a full lap, one OppWpnt per waypoint.
#
# Field names, types and order are copied verbatim from ForzaETH's
# f110_msgs/OpponentTrajectory.
#
# lap_count is how many laps of the opponent have been observed, float64 as in
# the original. Consumers treat lap_count < 1 as "trajectory not yet
# trustworthy".
#
# The oppwpnts are in increasing s. collision_prediction.py assumes the last
# point duplicates the first; publishers should keep it that way.

std_msgs/Header header
int32 id
float64 lap_count
bool opp_is_on_trajectory
OppWpnt[] oppwpnts
