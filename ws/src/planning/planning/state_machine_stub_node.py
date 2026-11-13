"""Publish a fixed state on /state_machine (std_msgs/String). A stand-in, not a state machine.

See state_machine_stub.py for which state does what to the shared ports. The
state is the ROS parameter `state` and can be changed while running:

    ros2 run planning state_machine_stub_node --ros-args -p state:=OVERTAKE
    ros2 param set /state_machine_stub state TRAILING

It is published at `rate` Hz, VOLATILE, as a state machine publishing its
current state every cycle would. Unknown states are refused at startup and
rejected by `ros2 param set`, so the published state is always a known one.
"""

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from planning.state_machine_stub import validate_state


class StateMachineStubNode(Node):

    def __init__(self, **kwargs):
        super().__init__('state_machine_stub', **kwargs)
        p = lambda n, v: self.declare_parameter(n, v).value

        self.state = validate_state(p('state', 'GB_TRACK'))
        topic = p('topic', '/state_machine')
        rate = float(p('rate', 10.0))
        if rate <= 0:
            raise RuntimeError(f'rate must be positive, got {rate}')

        self.add_on_set_parameters_callback(self.param_cb)
        self.pub = self.create_publisher(String, topic, 10)
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().warn(
            f'STAND-IN state machine: publishing the fixed state {self.state} on {topic} at '
            f'{rate:.0f} Hz. It never changes state by itself and ignores force_trailing.')

    def param_cb(self, params):
        for prm in params:
            if prm.name == 'state':
                try:
                    validate_state(prm.value)
                except ValueError as e:
                    return SetParametersResult(successful=False, reason=str(e))
            elif prm.name in ('topic', 'rate'):
                return SetParametersResult(successful=False,
                                           reason=f'{prm.name} is fixed at startup')
        for prm in params:
            if prm.name == 'state' and prm.value != self.state:
                self.get_logger().info(f'state {self.state} -> {prm.value}')
                self.state = prm.value
        return SetParametersResult(successful=True)

    def tick(self):
        self.pub.publish(String(data=self.state))


def main(args=None):
    rclpy.init(args=args)
    node = StateMachineStubNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
