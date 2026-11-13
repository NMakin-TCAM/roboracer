"""The states the stand-in state machine may publish. Imports no ROS.

A STAND-IN, NOT A STATE MACHINE. Upstream's state machine decides the state from
the opponent, the collision prediction (collision_prediction/force_trailing) and
the local planners. This publishes whatever it is told to, so a run or a test can
put the graph in each state on purpose. Nothing reads force_trailing, and nothing
here ever changes state by itself.

What the shared ports do with /state_machine (std_msgs/String, by their code):

    update_waypoints_node     GB_TRACK held for hysteresis_time -> starts writing
                              measured speeds into the waypoints. Any other state
                              stops it. Never publishes an update before the first
                              GB_TRACK message (docs/forzaeth-quirks.md section 4).
    opponent_trajectory_node  OVERTAKE -> /opponent_trajectory messages are not
                              stored. Every other state is the same to it.
    collision_prediction_node stores the state and never reads it.
    predictor_opponent_...    does not subscribe.

So GB_TRACK, TRAILING and OVERTAKE between them reach every state-gated branch of
the four ports. Upstream has more states; add them here when a consumer reads one.
A typo such as 'overtake' would leave both gated nodes silently inert, so unknown
names are refused rather than published.
"""

KNOWN_STATES = ('GB_TRACK', 'TRAILING', 'OVERTAKE')


def validate_state(state):
    """Return state if it is one the stand-in may publish, else raise ValueError."""
    if state not in KNOWN_STATES:
        raise ValueError(f'unknown state {state!r}; the stand-in publishes one of '
                         f'{", ".join(KNOWN_STATES)} (case matters)')
    return state
