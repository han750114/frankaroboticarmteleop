import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from frankx import Robot, LinearRelativeMotion, Affine
import threading
import numpy as np
import time

ROBOT_IP = "172.16.0.2"
WRIST_Z_NEUTRAL = 0.28
WRIST_Z_HIGH = 0.3
MOVE_STEP = 0.01  # 每次移動 1cm
MOVE_INTERVAL = 0.2  # 每次間隔時間（秒）

class FrankaSteadyZControl(Node):
    def __init__(self):
        super().__init__('franka_steady_z_control')
        self.get_logger().info("🧪 Franka Steady Z Control 啟動中（閾值控制高度）")

        self.robot = Robot(ROBOT_IP)
        self.robot.set_default_behavior()
        self.robot.set_dynamic_rel(0.15)

        self.lock = threading.Lock()
        self.current_mode = "idle"  # idle, up, down

        # 訂閱手腕位置
        self.create_subscription(Pose, '/wrist', self.wrist_callback, 10)

        # 啟動持續移動執行緒
        self.keep_moving = True
        self.movement_thread = threading.Thread(target=self.movement_loop, daemon=True)
        self.movement_thread.start()

    def wrist_callback(self, msg: Pose):
        wrist_z = msg.position.z

        abs_z = abs(wrist_z)

        if abs_z > WRIST_Z_HIGH:
            self.current_mode = "up"
        elif abs_z < WRIST_Z_NEUTRAL and abs_z!=0:
            self.current_mode = "down"
        else:
            self.current_mode = "idle"

        print(f"[🖐️ Wrist Z] {wrist_z:.3f} → Mode: {self.current_mode}")

    def movement_loop(self):
        while self.keep_moving:
            with self.lock:
                if self.current_mode == "up":
                    motion = LinearRelativeMotion(Affine(0.0, 0.0, MOVE_STEP))
                    self.robot.move(motion)
                    print("⬆️ Steady UP")
                elif self.current_mode == "down":
                    motion = LinearRelativeMotion(Affine(0.0, 0.0, -MOVE_STEP))
                    self.robot.move(motion)
                    print("⬇️ Steady DOWN")
                else:
                    # Idle mode
                    time.sleep(MOVE_INTERVAL)
                    continue

            time.sleep(MOVE_INTERVAL)

    def destroy_node(self):
        self.keep_moving = False
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = FrankaSteadyZControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🔚 關閉 Franka Steady Z Control")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
