import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray,Float32
from argparse import ArgumentParser
from frankx import *
from time import sleep
import numpy as np

class DisplacementSubscriber(Node):
    def __init__(self, robot):
        super().__init__('displacement_subscriber')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/displacement',
            self.listener_callback,
            10)
        self.subscription

        self.finger_subscription = self.create_subscription(
            Float32,
            '/finger',
            self.listener_callback_finger,
            10)
        self.subscription

        self.robot = robot  
        self.delta_range = 0.05  
        #self.initial_position = [ 0.3, 0, 0.5]
        self.initial_position = [0.4, 0.09, 0.26]
        
        self.left_qpos = 0.0  # 預設手指為開的狀態
        self.last_gripper_state = None

        self.gripper = self.robot.get_gripper()
        #self.gripper.homing()#gripper init 

        joint_motion = JointMotion([0.0053, -0.312 ,  0.2176, -2.5857,  0.0787,  2.2788,  1.8525])
        robot.move(joint_motion)

        #self.gripper = self.robot.get_gripper()
        self.impedance_motion = ImpedanceMotion(200.0, 20.0) 
        self.robot_thread = self.robot.move_async(self.impedance_motion)
        sleep(0.1)

        self.initial_target = self.impedance_motion.target
        self.get_logger().info(f'Initial target: {self.initial_target}')

    def listener_callback_finger(self, msg):
        if msg.data != 0:
            new_qpos = msg.data
            if abs(new_qpos - self.left_qpos) > 0.01:  # Only act on significant change
                self.left_qpos = new_qpos
                self.get_logger().info(f'Finger data received: {self.left_qpos}')
                # Move gripper in a separate thread so it doesn't bloc
                from threading import Thread
                Thread(target=self.gripper.move_async, args=(self.left_qpos * 0.5,)).start()
        else:
            self.get_logger().warn('Received finger does not have values.')


    def listener_callback(self, msg):
        displacement = msg.data
        if len(displacement) == 6:
            x, y, z, roll, pitch, yaw= displacement
            self.apply_relative_motion( x, y, z, pitch, yaw)
        else:
            self.get_logger().warn('Received displacement does not have exactly 3 values.')

    # def finger_status_open(self):

    #     print("finger position:", self.left_qpos)
    #     if left_finger_position < 0.8:#threshold
    #         return True
    #     else:
    #         return False

    def apply_relative_motion(self, delta_x, delta_y, delta_z, pitch, yaw):
        new_x = self.initial_position[0] + delta_x
        new_y = self.initial_position[1] + delta_y
        new_z = self.initial_position[2] + delta_z
        # Convert units
        pitch = pitch / 10 * 17.4
        yaw = yaw / 10 * 17.4
        #self.gripper.move(self.left_qpos*0.5)#self.gripper.move(0.08)

        # current_state = self.finger_status_open()
        # print("finger open:", current_state)

        # if current_state != self.last_gripper_state:
        #     if current_state:
        #         self.get_logger().info("Left hand fingers are open, Gripper releasing.")
        #         self.gripper.move(0.08)#self.gripper.move(0.08)
        #     else:
        #         self.get_logger().info("Left hand fingers are closed, Gripper clamping.")
        #         self.gripper.move(0.0)#self.gripper.move(0)
        #     self.last_gripper_state = current_state

        if not (0.25 <= new_x <= 0.43):
            self.get_logger().warn(f'X value {new_x:.4f} out of range (0.25 - 0.43). Motion ignored.')
            return
        if not (-0.35 <= new_y <= 0.34):
            self.get_logger().warn(f'Y value {new_y:.4f} out of range (-0.35 - 0.34). Motion ignored.')
            return
        if not (new_z <= 0.71):
            self.get_logger().warn(f'Z value {new_z:.4f} out of range (must be <= 0.71). Motion ignored.')
            return

        if hasattr(self, 'impedance_motion'):
            self.impedance_motion.target = Affine(new_x, new_y, new_z, yaw, pitch, 0)
            sleep(1/120)
            self.get_logger().info(f'Robot moved: dx={new_x:.4f}, dy={new_y:.4f}, dz={new_z:.4f}, roll=0, pitch={pitch}')
        else:
            self.get_logger().error('Impedance motion is not initialized!')


    def stop_motion(self):
        if hasattr(self, 'impedance_motion'):
            self.impedance_motion.finish()
            self.robot_thread.join()
            self.get_logger().info("Impedance motion finished.")

def main(args=None):


    rclpy.init(args=args)

    parser = ArgumentParser()
    parser.add_argument('--host', default='172.16.0.2', help='FCI IP of the robot')
    args = parser.parse_args()

    robot = Robot(args.host)

    robot.set_default_behavior()
    robot.recover_from_errors()
    robot.set_dynamic_rel(0.15) 

    
    subscriber = DisplacementSubscriber(robot)
    try:
        rclpy.spin(subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.stop_motion() 
        subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()