import cv2
import easyocr
import kinematics as kin
import numpy as np
from camera import Camera
from MyRobot import MyRobot

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_ASPECT = 2
MIN_CONF = 0.95


reader = easyocr.Reader(["en"], gpu=False)


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
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop = True

    print("Enabling motors and moving to home then last read position...")

    robot.enable_motors()
    robot.move_j(0, 0, 0, 0)
    robot.move_j(*q)


def read_keys(img, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    keys = {}

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if h / w > MAX_ASPECT or w / h > MAX_ASPECT:
            continue

        crop = binary[y : y + h, x : x + w]

        # invert to black-on-white, pad, upscale ~4x for the recognizer
        padded = cv2.copyMakeBorder(
            255 - crop, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255
        )
        padded = cv2.resize(padded, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        big_h, big_w = padded.shape
        result = reader.recognize(
            padded,
            horizontal_list=[[0, big_w - 1, 0, big_h - 1]],
            free_list=[],
            allowlist=ALLOWLIST,
            detail=1,
        )

        if not result or len(result) != 1:
            continue

        text, conf = result[0][1], result[0][2]

        if len(text) != 1 or conf < MIN_CONF:
            continue

        if text in keys:
            keys.pop(text, None)
            continue

        keys[text] = conf

    return keys


def init_ocr_thresh(img):
    best_theshold = None
    best_count = -1
    best_conf_sum = -1

    for threshold in range(100, 251, 5):
        keys = read_keys(img, threshold)

        n_keys = len(keys)
        conf_sum = sum(keys.values())

        if n_keys > best_count or (n_keys == best_count and conf_sum > best_conf_sum):
            best_theshold = threshold
            best_count = n_keys
            best_conf_sum = conf_sum

        if n_keys > 0:
            print(
                f"Threshold {threshold}: {n_keys} keys detected with avg conf {conf_sum / n_keys}"
            )
        else:
            print(f"Threshold {threshold}: {n_keys} keys detected")

    print(
        f"Best threshold: {best_theshold} ({best_count} keys detected with avg conf {best_conf_sum / best_count})"
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
        robot = MyRobot(device_name="/dev/ttyACM0")

        if not robot.init_status:
            print("Robot initialization failed. Exiting...")
            return

    cam = Camera(index=0, api_pref=cv2.CAP_V4L2)

    try:
        img = None

        if find_pos:
            init_robot_pos(robot, cam)
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
