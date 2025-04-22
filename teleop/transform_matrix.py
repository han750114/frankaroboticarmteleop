import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R
import numpy as np
from frankx import Robot

class WristPoseTransformer(Node):
    def __init__(self):
        super().__init__('wrist_pose_transformer')
        self.subscription_pose = self.create_subscription(
            Pose,
            '/left_wrist_pose',
            self.left_wrist_pose_callback,
            10)

        self.subscription_distance = self.create_subscription(
            Float64,
            '/left_thumb_index_distance',
            self.left_thumb_index_callback,
            10)

        self.latest_pose = None
        self.latest_distance = None
        self.pose_received = False
        self.distance_received = False

        self.robot = Robot("172.16.0.2")

        self.transforms = []
        self.max_samples = 10

        self.get_logger().info("Waiting for both topics...")

    def left_thumb_index_callback(self, msg):
        self.latest_distance = msg.data
        self.distance_received = True
        self.try_compute()

    def left_wrist_pose_callback(self, msg):
        self.latest_pose = msg
        self.pose_received = True
        self.try_compute()

    def try_compute(self):
        if self.pose_received and self.distance_received and len(self.transforms) < self.max_samples:
            self.get_logger().info(f"[{len(self.transforms)+1}/{self.max_samples}] Receiving data and computing transform...")

            vision_T_wrist = self.pose_to_matrix(self.latest_pose)

            state = self.robot.read_once()
            qpos = state.q
            robot_T_wrist = self.robot.kinematics().forward(qpos)

            T_robot_base_to_vision = robot_T_wrist @ np.linalg.inv(vision_T_wrist)
            self.transforms.append(T_robot_base_to_vision)

            if len(self.transforms) == self.max_samples:
                self.compute_averaged_transform()

    def pose_to_matrix(self, pose):
        pos = pose.position
        ori = pose.orientation
        matrix = np.eye(4)
        matrix[:3, :3] = R.from_quat([ori.x, ori.y, ori.z, ori.w]).as_matrix()
        matrix[:3, 3] = [pos.x, pos.y, pos.z]
        return matrix

    def compute_averaged_transform(self):
        translations = np.array([t[:3, 3] for t in self.transforms])
        rotations = np.array([R.from_matrix(t[:3, :3]).as_quat() for t in self.transforms])

        avg_translation = np.mean(translations, axis=0)
        avg_rotation = R.from_quat(rotations).mean().as_matrix()  # Robust mean rotation

        avg_matrix = np.eye(4)
        avg_matrix[:3, :3] = avg_rotation
        avg_matrix[:3, 3] = avg_translation

        self.get_logger().info("✅ Averaged Transform (Vision Pro → Robot Base):\n" + str(np.round(avg_matrix, 4)))

        rclpy.shutdown()  # Cleanly exit


def main(args=None):
    rclpy.init(args=args)
    node = WristPoseTransformer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
