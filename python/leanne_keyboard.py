import time

import cv2
import easyocr
import kinematics as kin
import numpy as np
from camera import Camera
from MyRobot import MyRobot

# TODO: find keyboard viewing position
VIEW_POS = (  7.47,   16.85,  -74.71, -103.6)

KEY_W, KEY_H = 15, 15

KEY_OFFSET_X, KEY_OFFSET_Y = 5, -5

HOVER_OFFSET_Z = 10
PRESS_OFFSET_Z = -5

STYLUS_DOWN_PITCH = 180

Q_X = 0
A_X = 4
Z_X = 15

SPACE_ROW_Y = -1 * KEY_H

TOP_ROW_Z = 0
MID_ROW_Z = 0
BOT_ROW_Z = 0
SPACE_ROW_Z = 0

TOP_ROW_CHARS = "QWERTYUIOP"
MID_ROW_CHARS = "ASDFGHJKL"
BOT_ROW_CHARS = "ZXCVBNM"

IMAGE_PATH = "python/keyboard_images/keyboard_20261029_103154_raw.png"

BINARY_THRESHOLD = 195
ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_ASPECT = 2
MIN_CONF = 0.95

PHRASE = "hello world"

H_5C = np.array(
    [[0, 0, 1, 0],
     [0, -1, 0, 0],
     [1, 0, 0, 0],
     [0, 0, 0, 1]], dtype=float
)

reader = easyocr.Reader(["en"], gpu=False)

camera_matrix = np.array(
    [
        [662.4474231057976, 0, 350.98189014851386],
        [0, 660.071481472615, 222.31312503748862],
        [0, 0, 1],
    ]
)
dist = np.array([[-0.011341169941478515, 0.03010347390242609, 0, 0, 0]])


# keyboard frame's origin is at the bottom left corner
# x axis starts at the left edge of the 'Q' key
# y axis starts at the top edge of the 'Z' key
# z axis is perpendicular to the keyboard plane
def generate_keyboard_map():
    top_row_pos = [
        (Q_X + i * KEY_W, 2 * KEY_H, TOP_ROW_Z) for i in range(len(TOP_ROW_CHARS))
    ]

    middle_row_pos = [
        (A_X + i * KEY_W, 1 * KEY_H, MID_ROW_Z) for i in range(len(MID_ROW_CHARS))
    ]

    bottom_row_pos = [
        (Z_X + i * KEY_W, 0 * KEY_H, BOT_ROW_Z) for i in range(len(BOT_ROW_CHARS))
    ]

    b_index = BOT_ROW_CHARS.index("B")
    b_x = bottom_row_pos[b_index][0]

    keyboard_map = {
        **dict(zip(TOP_ROW_CHARS, top_row_pos)),
        **dict(zip(MID_ROW_CHARS, middle_row_pos)),
        **dict(zip(BOT_ROW_CHARS, bottom_row_pos)),
        " ": (b_x, SPACE_ROW_Y, SPACE_ROW_Z),
    }

    return keyboard_map


def read_keys(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    debug = img.copy()
    cv2.drawContours(debug, contours, -1, (0, 255, 0), 2)
    cv2.imshow("contours", debug)
    cv2.waitKey(1)

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
            # remove duplicate keys
            keys.pop(text, None)
            continue

        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255),1)

        keys[text] = (x, y)
    
    cv2.imshow("h", img)
    cv2.waitKey(100)
    return keys


def get_cam_keyboard_H(detected_keys, keyboard_map):
    obj_pts = []
    img_pts = []

    for k, v in detected_keys.items():
        obj_pts.append(keyboard_map[k])
        img_pts.append(v)

    obj_pts = np.array(obj_pts, dtype=np.float32)
    img_pts = np.array(img_pts, dtype=np.float32)

    ok, rvec, tvec, _ = cv2.solvePnPRansac(obj_pts, img_pts, camera_matrix, dist)

    if not ok:
        raise RuntimeError("Could not solve PnP for keyboard pose")

    R, _ = cv2.Rodrigues(rvec)

    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = tvec.flatten()

    return H


def get_key_in_cam_frame(H_ck, key, keyboard_map):
    x, y, z = keyboard_map[key]
    pos_k = np.array([x + KEY_OFFSET_X, y + KEY_OFFSET_Y, z, 1]).reshape(4, 1)
    pos_c = H_ck @ pos_k
    return pos_c[:3].flatten()


def get_keys_in_cam_frame(H_ck, keyboard_map):
    cam_frame_map = {}

    for k in keyboard_map:
        cam_frame_map[k] = get_key_in_cam_frame(H_ck, k, keyboard_map)

    return cam_frame_map


def get_key_in_robot_frame(H_ck, key, keyboard_map, q_deg):
    # keyboard pose in the robot base frame
    H_rk = kin.direct_kinematics(np.radians(q_deg), frame=5) @ H_5C @ H_ck

    x, y, z = keyboard_map[key]
    pos_k = np.array([x + KEY_OFFSET_X, y + KEY_OFFSET_Y, z, 1]).reshape(4, 1)
    pos_r = H_rk @ pos_k

    return pos_r[:3].flatten()


def get_keys_in_robot_frame(H_ck, keyboard_map, q_deg):
    robot_frame_map = {}

    for k in keyboard_map:
        robot_frame_map[k] = get_key_in_robot_frame(H_ck, k, keyboard_map, q_deg)

    return robot_frame_map


def draw_target_key(img, H_ck, key, keyboard_map):
    x, y, z = keyboard_map[key]

    # corners of the target box, followed by the aim point
    pts_k = np.array(
        [
            [x, y, z, 1],
            [x + 2 * KEY_OFFSET_X, y, z, 1],
            [x + 2 * KEY_OFFSET_X, y + 2 * KEY_OFFSET_Y, z, 1],
            [x, y + 2 * KEY_OFFSET_Y, z, 1],
            [x + KEY_OFFSET_X, y + KEY_OFFSET_Y, z, 1],
        ],
        dtype=np.float64,
    )

    # points are already in the camera frame, so rvec and tvec are zero
    pts_c = (H_ck @ pts_k.T).T[:, :3]
    pts_px, _ = cv2.projectPoints(
        pts_c, np.zeros(3), np.zeros(3), camera_matrix, dist
    )
    pts_px = pts_px.reshape(-1, 2)

    x0, y0 = np.round(pts_px[:4].min(axis=0)).astype(int).tolist()
    x1, y1 = np.round(pts_px[:4].max(axis=0)).astype(int).tolist()
    u, v = np.round(pts_px[4]).astype(int).tolist()

    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.drawMarker(img, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)

    return u, v

def main():
    cam = Camera()
    robot = MyRobot()
    print(kin.direct_kinematics(np.radians(VIEW_POS)))
    if not robot.init_status:
        print("Robot initialization failed. Exiting...")
        return

    try:
        robot.move_j(0, 0, 0, 0)

        keyboard_map = generate_keyboard_map()

        phrase = input(f"Enter phrase to type (default: '{PHRASE}'): ") or PHRASE
        phrase = phrase.strip().upper()

        for key in phrase:
            if key not in keyboard_map:
                print(f"Key '{key}' not found in keyboard map. Skipping...")
                continue

            print(f"Key: {key}")

            robot.move_j(*VIEW_POS)
            time.sleep(1)
            # img = cv2.imread(IMAGE_PATH)
            cam.read()
            img = cam.read()
            keys = read_keys(img)

            if len(keys) < 3:
                print("Not enough keys detected. Skipping...")
                continue

            print(f"Detected {len(keys)} keys")
            
            H_ck = get_cam_keyboard_H(keys, keyboard_map)

            key_cam_pos = get_key_in_cam_frame(H_ck, key, keyboard_map)

            target_px = draw_target_key(img, H_ck, key, keyboard_map)
            print(f"Reprojected pixel coords: {target_px}")
            cv2.imshow("h", img)
            cv2.waitKey(100)

            q_deg = robot.read_joint_angles()

            q_deg = robot.read_joint_angles()
            key_pos = get_key_in_robot_frame(H_ck, key, keyboard_map, q_deg)

            hover_pos = key_pos + np.array([0, 0, HOVER_OFFSET_Z])
            press_pos = key_pos + np.array([0, 0, PRESS_OFFSET_Z])

            print(f"Pixel coords: {keys.get(key, 'not detected')}")
            print(f"Camera frame coords: {key_cam_pos}")
            print(f"Robot frame coords: {key_pos}")
            print(f"Hover pos: {hover_pos}")
            print(f"Press pos: {press_pos}")
            print()

            robot.move_c(*hover_pos, STYLUS_DOWN_PITCH)
            time.sleep(0)
            robot.move_c(*press_pos, STYLUS_DOWN_PITCH)
            time.sleep(0)
            robot.move_c(*hover_pos, STYLUS_DOWN_PITCH)
            time.sleep(0)
    finally:
        cam.cap.release()
        cv2.destroyAllWindows()
        robot.disable_motors()


if __name__ == "__main__":
    main()
