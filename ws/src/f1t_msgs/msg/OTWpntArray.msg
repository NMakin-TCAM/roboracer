# An overtaking trajectory produced by a local planner.
#
# An EMPTY wpnts array is the planner saying "no overtake this cycle": either
# nothing is in the way, or no feasible evasion exists and the controller
# should keep tracking the global raceline. Publish every cycle so a consumer
# can tell that apart from a dead planner.
#
# ot_side is "left", "right" or "" when not overtaking. side_switch is true on
# the cycle the side flips, which is the event worth alarming on: a planner
# that flips sides repeatedly is worse than one that picks badly and commits.
std_msgs/Header header
builtin_interfaces/Time last_switch_time
bool side_switch
string ot_side
string ot_line
Wpnt[] wpnts
