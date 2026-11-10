import cv2
import kinematics as kin
import numpy as np
from camera import Camera
from MyRobot import MyRobot


def init_robot_pos(robot, cam):
    robot.move_j(0, 0, 0, 0)

    # Torque OFF on every servo, but keep the serial port open so we can still read.
    for dxl_id in robot.motor_ids:
        robot._write1(dxl_id, robot.ADDR_MX_TORQUE_ENABLE, robot.TORQUE_DISABLE)
    print("Torque released. You can now move the arm by hand.")

    stop = False

    while not stop:
        q = robot.read_joint_angles()

        xyz4 = kin.direct_kinematics(np.radians(q), frame=4)[0:3, 3]  # stylus joint
        xyz5 = kin.direct_kinematics(np.radians(q), frame=5)[0:3, 3]  # camera frame

        print("Joint angles (deg):", q)
        print("Stylus xyz (mm)   :", xyz4)
        print("Camera xyz (mm)   :", xyz5)
        print()

        img = cam.read()
        cv2.imshow("camera view", img)
        if cv2.waitKey(1000) & 0xFF == ord("q"):
            stop = True

    print("Enabling motors and moving to home then last read position...")

    robot.enable_motors()
    robot.move_j(0, 0, 0, 0)
    robot.move_j(*q)


def init_ocr_thresh(img):
    best_threshold = None
    best_count = -1
    best_conf_sum = -1

    for threshold in range(100, 251, 5):
        ocr_keys = Camera.ocr_keys(img, threshold, draw=True)
        keys = {k: v[0] for k, v in ocr_keys.items()}

        n_keys = len(keys)
        conf_sum = sum(keys.values())

        if n_keys > best_count or (n_keys == best_count and conf_sum > best_conf_sum):
            best_threshold = threshold
            best_count = n_keys
            best_conf_sum = conf_sum

        log_statement = f"Threshold {threshold}: {n_keys} keys detected"
        if n_keys > 0:
            log_statement += f" with avg conf {conf_sum / n_keys}"
        print(log_statement)

    if best_count < 1:
        print("No keys detected at any threshold")
        return

    print(
        f"Best threshold: {best_threshold} ({best_count} keys detected with avg conf {best_conf_sum / best_count})"
    )


def main():
    robot = None
    cam = None

    opt = int(
        input("""
Options:
    1) Find robot position
    2) Find best OCR threshold
    3) Both (1) and (2)
Enter your choice (1, 2, or 3): """)
    )

    find_pos = opt in [1, 3]
    find_thresh = opt in [2, 3]

    if not find_pos and not find_thresh:
        print("No valid option selected. Exiting...")
        return

    if find_pos:
        robot = MyRobot()

        if not robot.init_status:
            print("Robot initialization failed. Exiting...")
            return

    cam = Camera()

    try:
        img = None

        if find_pos:
            init_robot_pos(robot, cam)
            robot.wait_settled()
            img = cam.read()

        if find_thresh:
            if img is None:
                print("Move arm until keyboard is visible")

                stop = False

                while not stop:
                    img = cam.read()
                    cv2.imshow("camera view", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop = True

            init_ocr_thresh(img)

    except KeyboardInterrupt:
        pass
    finally:
        if robot is not None:
            print("\nGoing back to home...")
            robot.move_j(0, 0, 0, 0)
            robot.disable_motors()

        if cam is not None:
            cam.cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
