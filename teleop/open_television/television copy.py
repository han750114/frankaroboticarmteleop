import time
from vuer import Vuer
from vuer.schemas import ImageBackground, Hands
from multiprocessing import Array, Process, shared_memory
import numpy as np
import asyncio
import cv2
from vuer.schemas import Text

from multiprocessing import context
Value = context._default_context.Value


class TeleVision:
    def __init__(self, binocular, img_shape, img_shm_name, cert_file="/home/csl/avp_teleoperate/teleop/cert.pem", key_file="/home/csl/avp_teleoperate/teleop/key.pem", ngrok=False):
        print(f"[TeleVision Init] img_shape={img_shape}, types={[type(d) for d in img_shape]}")
        self.binocular = binocular
        self.img_height = img_shape[0]
        if binocular:
            print(f"[DEBUG] img_shape[1] type: {type(img_shape[1])}, value: {img_shape[1]}")
            self.img_width  = img_shape[1] // 2
        else:
            self.img_width  = img_shape[1]

        if ngrok:
            self.vuer = Vuer(host='0.0.0.0', queries=dict(grid=False), queue_len=3)
        else:
            if cert_file is None:
                print("cert_file is None")
            self.vuer = Vuer(host='0.0.0.0', cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)

        self.vuer.add_handler("HAND_MOVE")(self.on_hand_move)
        self.vuer.add_handler("CAMERA_MOVE")(self.on_cam_move)

        existing_shm = shared_memory.SharedMemory(name=img_shm_name)
        self.img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=existing_shm.buf)

        if binocular:
            self.vuer.spawn(start=False)(self.main_image_binocular)
        else:
            self.vuer.spawn(start=False)(self.main_image_monocular)

        self.left_hand_shared = Array('d', 16, lock=True)
        self.right_hand_shared = Array('d', 16, lock=True)
        self.left_landmarks_shared = Array('d', 75, lock=True)
        self.right_landmarks_shared = Array('d', 75, lock=True)
        
        self.head_matrix_shared = Array('d', 16, lock=True)
        self.aspect_shared = Value('d', 1.0, lock=True)

        self.process = Process(target=self.vuer_run)
        self.process.daemon = True
        self.process.start()

    
    def vuer_run(self):
        #self.vuer.run()
        self.vuer.run(async_start=True)  # async loop

    async def on_cam_move(self, event, session, fps=60):
        try:
            self.head_matrix_shared[:] = event.value["camera"]["matrix"]
            self.aspect_shared.value = event.value['camera']['aspect']
        except:
            pass

    async def on_hand_move(self, event, session, fps=60):
        try:
            left_landmarks = np.array(event.value["leftLandmarks"]).reshape(25, 3)
            right_landmarks = np.array(event.value["rightLandmarks"]).reshape(25, 3)

            self.left_hand_shared[:] = event.value["leftHand"]
            self.right_hand_shared[:] = event.value["rightHand"]
            self.left_landmarks_shared[:] = left_landmarks.flatten()
            self.right_landmarks_shared[:] = right_landmarks.flatten()

            # thumb to index distance
            left_distance = np.linalg.norm(left_landmarks[4] - left_landmarks[8])
            right_distance = np.linalg.norm(right_landmarks[4] - right_landmarks[8])
            print(f"Left hand thumb–index distance: {left_distance:.3f} m")
            print(f"Right hand thumb–index distance: {right_distance:.3f} m")

            # Wrist pos and orientation
            left_hand_matrix = np.array(event.value["leftHand"]).reshape(4, 4, order="F")
            right_hand_matrix = np.array(event.value["rightHand"]).reshape(4, 4, order="F")

            left_pos = left_hand_matrix[:3, 3]
            right_pos = right_hand_matrix[:3, 3]

            from scipy.spatial.transform import Rotation as R
            left_quat = R.from_matrix(left_hand_matrix[:3, :3]).as_quat()
            right_quat = R.from_matrix(right_hand_matrix[:3, :3]).as_quat()

            print(f"Left Wrist Position: {left_pos}")
            print(f"Left Wrist Quaternion: {left_quat}")
            print(f"Right Wrist Position: {right_pos}")
            print(f"Right Wrist Quaternion: {right_quat}")

        except Exception as e: 
            print(f"[on_hand_move] error: {e}")

    
    async def main_image_binocular(self, session, fps=60):
        try:
            session.upsert @ Hands(fps=fps, stream=True, key="hands", showLeft=True, showRight=True)
            while True:
                display_image = cv2.cvtColor(self.img_array, cv2.COLOR_BGR2RGB)

                try:
                    left_lm = self.left_landmarks
                    right_lm = self.right_landmarks

                    left_dist = np.linalg.norm(left_lm[4] - left_lm[8])
                    right_dist = np.linalg.norm(right_lm[4] - right_lm[8])

                    font_scale = 1.0
                    thickness = 3
                    text_color_left = (255, 0, 0)
                    text_color_right = (0, 0, 255)

                    x = 30
                    y_center = self.img_height // 2

                    cv2.putText(display_image, f"L: {left_dist:.3f} m", (x, y_center - 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color_left, thickness)
                    cv2.putText(display_image, f"R: {right_dist:.3f} m", (x, y_center + 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color_right, thickness)

                except Exception as e:
                    print(f"Distance overlay error: {e}")

                # ---- Send annotated image ----
                session.upsert(
                    [
                        ImageBackground(
                            display_image[:, :self.img_width],
                            aspect=1.778,
                            height=1,
                            distanceToCamera=1,
                            layers=1,
                            format="jpeg",
                            quality=50,
                            key="background-left",
                            interpolate=True,
                        ),
                        ImageBackground(
                            display_image[:, self.img_width:],
                            aspect=1.778,
                            height=1,
                            distanceToCamera=1,
                            layers=2,
                            format="jpeg",
                            quality=50,
                            key="background-right",
                            interpolate=True,
                        ),
                    ],
                    to="bgChildren",
                )

                await asyncio.sleep(0.016 * 2)
        except AssertionError as e:
            print(f"[main_image_binocular] websocket dropped: {e}")
        except Exception as e:
            print(f"[main_image_binocular] unexpected error: {e}")



    # async def main_image_binocular(self, session, fps=60):
    #     session.upsert @ Hands(fps=fps, stream=True, key="hands", showLeft=False, showRight=False)
    #     while True:
    #         display_image = cv2.cvtColor(self.img_array, cv2.COLOR_BGR2RGB)
    #         # aspect_ratio = self.img_width / self.img_height
    #         session.upsert(
    #             [
    #                 ImageBackground(
    #                     display_image[:, :self.img_width],
    #                     aspect=1.778,
    #                     height=1,
    #                     distanceToCamera=1,
    #                     # The underlying rendering engine supported a layer binary bitmask for both objects and the camera. 
    #                     # Below we set the two image planes, left and right, to layers=1 and layers=2. 
    #                     # Note that these two masks are associated with left eye’s camera and the right eye’s camera.
    #                     layers=1,
    #                     format="jpeg",
    #                     quality=50,
    #                     key="background-left",
    #                     interpolate=True,
    #                 ),
    #                 ImageBackground(
    #                     display_image[:, self.img_width:],
    #                     aspect=1.778,
    #                     height=1,
    #                     distanceToCamera=1,
    #                     layers=2,
    #                     format="jpeg",
    #                     quality=50,
    #                     key="background-right",
    #                     interpolate=True,
    #                 ),
    #             ],
    #             to="bgChildren",
    #         )
    #         # 'jpeg' encoding should give you about 30fps with a 16ms wait in-between.
    #         await asyncio.sleep(0.016 * 2)

    async def main_image_monocular(self, session, fps=60):
        session.upsert @ Hands(fps=fps, stream=True, key="hands", showLeft=False, showRight=False)
        while True:
            display_image = cv2.cvtColor(self.img_array, cv2.COLOR_BGR2RGB)
            #print(display_image)
            # aspect_ratio = self.img_width / self.img_height
            session.upsert(
                [
                    ImageBackground(
                        display_image,
                        aspect=1.778,
                        height=1,
                        distanceToCamera=1,
                        format="jpeg",
                        quality=50,
                        key="background-mono",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            await asyncio.sleep(0.016)

    @property
    def left_hand(self):
        return np.array(self.left_hand_shared[:]).reshape(4, 4, order="F")
        
    
    @property
    def right_hand(self):
        return np.array(self.right_hand_shared[:]).reshape(4, 4, order="F")
        
    
    @property
    def left_landmarks(self):
        return np.array(self.left_landmarks_shared[:]).reshape(25, 3)
    
    @property
    def right_landmarks(self):
        return np.array(self.right_landmarks_shared[:]).reshape(25, 3)

    @property
    def head_matrix(self):
        return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")

    @property
    def aspect(self):
        return float(self.aspect_shared.value)
    
if __name__ == '__main__':
    import os 
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    import threading
    from image_server.image_client import ImageClient

    # image
    img_shape = (480, 640  , 3) #img_shape = (480, 640 * 2, 3)
    img_shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
    img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=img_shm.buf)
    img_client = ImageClient(tv_img_shape = img_shape, tv_img_shm_name = img_shm.name)
    image_receive_thread = threading.Thread(target=img_client.receive_process, daemon=True)
    image_receive_thread.start()

    # television
    tv = TeleVision(True, img_shape, img_shm.name)
    print("vuer unit test program running...")
    print("you can press ^C to interrupt program.")
    while True:
        time.sleep(0.03)