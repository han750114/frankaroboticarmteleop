import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Pose
from frankx import Robot, Gripper, JointMotion, MotionData, LinearRelativeMotion, Affine
import numpy as np
import threading
import time

ROBOT_IP = "172.16.0.2"  # 替換成你的 Franka IP
JOINT7_STEP = 0.05       # 每次控制 joint7 的旋轉角度（弧度）
WRIST_X_NEUTRAL = 0.1
WRIST_Y_NEUTRAL = 0.5
WRIST_Z_NEUTRAL = 0.0
DEADZONE = 0.02          # 避免太小的抖動也觸發旋轉


class FrankaTeleop(Node):
    def __init__(self):
        super().__init__('franka_teleop')
        self.get_logger().info("Franka Teleop Controller 啟動中")

        # 初始化 Franka
        self.robot = Robot(ROBOT_IP)
        self.robot.set_default_behavior()
        self.robot.set_dynamic_rel(0.15)
        self.gripper = Gripper(ROBOT_IP)

        self.last_joint_positions = self.robot.read_once().q
        self.lock = threading.Lock()
        self.wrist_neutral = None
        self.finger_updated = False
        self.wrist_updated = False

        # 建立訂閱者
        self.create_subscription(Float32, '/finger', self.finger_callback, 10)
        self.create_subscription(Pose, '/wrist', self.wrist_callback, 10)

    def finger_callback(self, msg: Float32):
        distance = msg.data
        self.finger_updated = True
        try:
            gripper_width = max(0.0, min(distance * 0.9, 0.08))  
            self.gripper.move(gripper_width)
            print(f"Gripper 寬度設定為: {gripper_width:.3f} m")
        except Exception as e:
            self.get_logger().error(f"Gripper 操作失敗: {e}")

    def wrist_callback(self, msg: Pose):
        self.wrist_updated = True
        try:
            current = np.array([msg.position.x, msg.position.y, msg.position.z])
            if self.wrist_neutral is None:
                self.wrist_neutral = np.array([msg.position.x, msg.position.y, msg.position.z])
                return
            
            delta = current - self.wrist_neutral

            # Don't move if the change is too small (i.e., "idle" input)
            if np.linalg.norm(delta) < DEADZONE:
                return
            
            x = msg.position.x
            y = msg.position.y
            z = msg.position.z

            dx = x - WRIST_X_NEUTRAL
            dy = y - WRIST_Y_NEUTRAL
            dz = z  # 如果你有設定 WRIST_Z_NEUTRAL，也可用 dz = z - WRIST_Z_NEUTRAL

            # 設定 DEADZONE 避免小抖動
            if abs(dx) < DEADZONE-0.02: dx = 0.0
            if abs(dy) < DEADZONE: dy = 0.0
            if abs(dz) < DEADZONE: dz = 0.0

            if dx != 0.0 or dy != 0.0 or dz != 0.0:
                with self.lock:
                    # 縮放移動大小（這裡用 0.05 代表手部動一下就對應 5cm 移動）
                    scale = 0.05
                    relative_affine = Affine(dx * scale, dy * scale, dz * scale)

                    motion = LinearRelativeMotion(relative_affine)
                    print(f"移動量: x={dx:.3f}, y={dy:.3f}, z={dz:.3f}")
                    self.robot.move(motion)
                
                # reset flags after move
                self.finger_updated = False
                self.wrist_updated = False

        except Exception as e:
            self.get_logger().error(f"末端運動控制失敗: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = FrankaTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("關閉 Franka Teleop")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
