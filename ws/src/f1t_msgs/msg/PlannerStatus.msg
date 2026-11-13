# What a local planner decided this cycle, published beside its OTWpntArray.
#
# Not a ForzaETH message: OTWpntArray is kept identical to theirs, and an empty
# OTWpntArray means two opposite things ("nothing in the way" and "in the way,
# cannot get round"). This says which.
#
# header.stamp is EXACTLY the stamp of the OTWpntArray it describes, so a
# consumer pairs the two by stamp and never acts on a status meant for another
# trajectory.
#
# status is one of (planning.spliner):
#   clear      nothing in the way; the trajectory is empty; follow the raceline
#   evading    the trajectory is the way round
#   no room    in the way, no side has room; the trajectory is empty; do NOT drive on
#   too close  in the way, too close to plan; the trajectory is empty; do NOT drive on

std_msgs/Header header
string status
string note
