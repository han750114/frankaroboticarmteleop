import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Pose
import numpy as np
from multiprocessing import shared_memory
import threading
import time
import asyncio

# 匯入你已有的類別
from open_television.television import TeleVision
from image_server.image_client import ImageClient
from scipy.spatial.transform import Rotation as R


class TeleVisionROSInterface(Node):
    def __init__(self, tv):
        super().__init__('television_ros_interface')
        self.tv = tv

        self.finger_pub = self.create_publisher(Float32, '/finger', 10)
        self.wrist_pub = self.create_publisher(Pose, '/wrist', 10)

        timer_period = 0.03  # 每 30 毫秒
        self.timer = self.create_timer(timer_period, self.publish_data)

    def publish_data(self):
        try:
            # 讀取手部 landmarks 與距離
            landmarks = self.tv.right_landmarks
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            distance = float(np.linalg.norm(thumb_tip - index_tip))

            finger_msg = Float32()
            finger_msg.data = distance
            self.finger_pub.publish(finger_msg)

            # 發佈手腕姿態
            hand_matrix = self.tv.right_hand
            position = hand_matrix[:3, 3]
            quat = R.from_matrix(hand_matrix[:3, :3]).as_quat()  # x, y, z, w

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
            self.get_logger().warn(f"Publish failed: {e}")


def main(args=None):
    rclpy.init(args=args)

    # --- 影像共享記憶體初始化 ---
    img_shape = (480, 640 , 3)
    shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
    img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=shm.buf)

    # --- 啟動攝影機影像接收 ---
    img_client = ImageClient(tv_img_shape=img_shape, tv_img_shm_name=shm.name)
    image_thread = threading.Thread(target=img_client.receive_process, daemon=True)
    image_thread.start()

    # --- 啟動 Vuer 顯示 + 手部追蹤 ---
    tv = TeleVision(binocular=True, img_shape=img_shape, img_shm_name=shm.name)

    # --- 啟動 ROS2 節點 ---
    ros_node = TeleVisionROSInterface(tv)

    print("TeleVision + ROS2 啟動成功：")
    print(" - 影像串流顯示中（使用 Vuer）")
    print(" - ROS2 topic：/finger, /wrist 正在發佈中")
    print("請使用 Ctrl+C 停止")

    try:
        rclpy.spin(ros_node)
    except KeyboardInterrupt:
        pass

    ros_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
