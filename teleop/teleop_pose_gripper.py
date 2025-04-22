import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from frankx import Gripper


class GripperControlNode(Node):
    def __init__(self):
        super().__init__('gripper_control_node')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/finger',
            self.finger_callback,
            10
        )
        self.subscription  # prevent unused variable warning

        self.gripper = Gripper("172.16.0.2")  # change to your Franka IP if needed
        self.prev_state = None
        self.threshold = 0.03  # threshold to determine close vs open

        self.get_logger().info("Gripper control node started, listening to /finger topic")

    def finger_callback(self, msg):
        if not msg.data or len(msg.data) < 1:
            self.get_logger().warn("Received empty finger data!")
            return

        distance = msg.data[0]
        self.get_logger().info(f"Finger distance: {distance:.4f} m")

        # If hand is closed (finger tip distance small), close gripper
        if distance < self.threshold:
            if self.prev_state != "closed":
                self.get_logger().info("Closing gripper")
                self.gripper.move(0.0)  # close
                self.prev_state = "closed"
        else:
            if self.prev_state != "open":
                self.get_logger().info("Opening gripper")
                self.gripper.move(0.08)  # open (max opening ~8cm)
                self.prev_state = "open"


def main(args=None):
    rclpy.init(args=args)
    node = GripperControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
