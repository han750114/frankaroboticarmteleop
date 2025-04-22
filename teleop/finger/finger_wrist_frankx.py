import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from frankx import Robot, Gripper
import threading

ROBOT_IP = "172.16.0.2"  # 替換成你的 Franka IP
MAX_GRIPPER_WIDTH = 0.08  # Max gripper opening in meters (adjust if needed)

class FrankaGripperTest(Node):
    def __init__(self):
        super().__init__('franka_gripper_test')
        self.get_logger().info("Franka Gripper Unit Test 啟動中（僅訂閱 /finger）")

        # 初始化 Franka gripper
        self.robot = Robot(ROBOT_IP)
        self.robot.set_default_behavior()
        self.robot.set_dynamic_rel(0.15)
        self.gripper = Gripper(ROBOT_IP)
        self.lock = threading.Lock()

        # 訂閱 /finger topic 控制 gripper 開口
        self.create_subscription(Float32, '/finger', self.finger_callback, 10)

    def finger_callback(self, msg: Float32):
        try:
            distance = msg.data
            gripper_width = max(0.0, min(distance * 0.9, MAX_GRIPPER_WIDTH))  # 限制開口寬度
            with self.lock:
                self.gripper.move(gripper_width)
            print(f"Gripper 控制成功：設定寬度 {gripper_width:.3f} m")
        except Exception as e:
            self.get_logger().error(f"Gripper 操作失敗: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = FrankaGripperTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("關閉 Franka Gripper Test")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
