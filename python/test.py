import time
import cv2
import numpy as np
from MyRobot import MyRobot
import kinematics as kin
from camera import Camera

POINTS = [
    (180,    0, 80,  90),   # 1: front, high, stylus horizontal
    (150,    0, 50,  90),   # 2: front, lower
    (150,    0, 30, 180),   # 3: stylus down (like pressing a key)
    (120,   80, 30, 180),   # 4: y positive (base rotation)
    (100, -100, 60, 135),   # 5: y negative, tilted stylus
]

robot = MyRobot()                                   # moves to (0, 0, 0, 0)
cam = Camera()
cam.cap.set(cv2.CAP_PROP_SETTINGS, 1)
if robot.init_status:
    try:
        for i, (x, y, z, pitch) in enumerate(POINTS, start=1):
            print(f"\n--- Point {i}: target ({x}, {y}, {z}), pitch {pitch} ---")
            robot.move_c(x, y, z, pitch)
            print("Joint angles (deg):", robot.read_joint_angles())
            print("read_ee_position  :", robot.read_ee_position()[0:3, 3])

            while cv2.waitKey(1) == -1:
                cv2.imshow("robot camera", cam.read())
            robot.move_j(0, 0, 0, 0)                # back to vertical (safe path)
    finally:
        cam.cap.release()
        cv2.destroyAllWindows()
        input("Hold the arm and press Enter to release torque...")
        robot.disable_motors()
