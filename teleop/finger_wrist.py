import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Pose
from multiprocessing import shared_memory
import threading
import numpy as np
import time

# 匯入你的 TeleVision 類別
from open_television.television import TeleVision


class TeleVisionROSInterface(Node):
    def __init__(self, tv):
        super().__init__('television_ros_interface')
        self.tv = tv

        self.finger_pub = self.create_publisher(Float32, '/finger', 10)
        self.wrist_pub = self.create_publisher(Pose, '/wrist', 10)

        # 開啟定時器，每 30ms 發佈一次
        timer_period = 0.03
        self.timer = self.create_timer(timer_period, self.publish_data)

    def publish_data(self):
        try:
            # 發佈 finger distance (左手)
            landmarks = self.tv.left_landmarks
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            dist = float(np.linalg.norm(thumb_tip - index_tip))

            finger_msg = Float32()
            finger_msg.data = dist
            self.finger_pub.publish(finger_msg)

            # 發佈 wrist pose
            hand_matrix = self.tv.left_hand
            position = hand_matrix[:3, 3]
            rotation_matrix = hand_matrix[:3, :3]

            from scipy.spatial.transform import Rotation as R
            quat = R.from_matrix(rotation_matrix).as_quat()  # x, y, z, w

            pose_msg = Pose()
            pose_msg.position.x = float(position[0])
            pose_msg.position.y = float(position[1])
            pose_msg.position.z = float(position[2])
            pose_msg.orientation.x = float(quat[0])
            pose_msg.orientation.y = float(quat[1])
            pose_msg.orientation.z = float(quat[2])
            pose_msg.orientation.w = float(quat[3])
            self.wrist_pub.publish(pose_msg)

        except Exception as e:
            self.get_logger().warn(f"Failed to publish data: {e}")


def main(args=None):
    rclpy.init(args=args)

    # 圖像共享記憶體初始化
    img_shape = (480, 640 * 2, 3)
    shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
    img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=shm.buf)

    # 啟動 Vuer client 影像接收（你應該已有 ImageClient 實作）
    from image_server.image_client import ImageClient
    img_client = ImageClient(tv_img_shape=img_shape, tv_img_shm_name=shm.name)
    image_thread = threading.Thread(target=img_client.receive_process, daemon=True)
    image_thread.start()

    # 啟動 television
    tv = TeleVision(binocular=True, img_shape=img_shape, img_shm_name=shm.name)

    # 啟動 ROS2 節點
    ros_node = TeleVisionROSInterface(tv)
    rclpy.spin(ros_node)

    ros_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
