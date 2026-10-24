import numpy as np
from MyRobot import MyRobot
import kinematics as kin

robot = MyRobot()                          # connects the robot
if robot.init_status:
    try:
        q_deg = robot.read_joint_angles()  # driver: Degrees
        q = np.radians(q_deg)              # Transforms form degrees to radians, it is always better to work with radians
        print(kin.direct_kinematics(q))              # Kinematics
    finally:
        robot.disable_motors()             # End