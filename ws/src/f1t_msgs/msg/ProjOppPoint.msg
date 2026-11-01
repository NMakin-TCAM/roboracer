# One observation of the opponent, projected onto the global raceline.
#
# Field names, types and order are copied verbatim from ForzaETH's
# f110_msgs/ProjOppPoint.
#
# CONVENTIONS, matching Obstacle.msg:
#
#   d > 0 is LEFT of the direction of travel, matching planning.frenet.FrenetConverter.
#
#   s is wrapped into [0, lap_length). Compare s values with
#   FrenetConverter.ds(), never by subtraction.
#
#   time is the observation time in seconds, as a plain float64 rather than a
#   builtin_interfaces/Time, as in the original.
#
#   The *_var fields are the variance of the OBSERVATION, not the physical
#   spread of the car. They must stay positive; the Gaussian process consumers
#   divide by them.

float64 s
float64 d
float64 vs                   # velocity along the reference line
float64 vd                   # velocity across it, positive to the left

bool is_static
bool is_visible              # seen at this time, as opposed to remembered

float64 time

float64 s_var
float64 d_var
float64 vs_var
float64 vd_var
