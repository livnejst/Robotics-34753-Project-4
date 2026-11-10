import os
from datetime import datetime

import cv2
import easyocr
import numpy as np
from config import Config

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_ASPECT = 2
MIN_AREA = 5 * 5
MIN_CONF = 0.9


camera_matrix = np.array(
    [
        [662.4474231057976, 0, 350.98189014851386],
        [0, 660.071481472615, 222.31312503748862],
        [0, 0, 1],
    ]
)
dist = np.array([[-0.011341169941478515, 0.03010347390242609, 0, 0, 0]])

reader = easyocr.Reader(["da"], gpu=False)


class Camera:
    def __init__(self, index=None):
        config = Config()

        self.index = (
            config.index if index is None else index
        )  # Index of the robot camera in my PC

        api_pref = cv2.CAP_DSHOW if config.is_win_os else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.index, api_pref)

        if not config.is_win_os:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        if not self.cap.isOpened():
            raise RuntimeError("CAMERA COULD NOT OPEN")

    def read(self):
        ok, frame = self.cap.read()

        if not ok:
            raise RuntimeError("COULD NOT READ FRAME")

        return frame

    @staticmethod
    def ocr_keys(img, thresh, draw=False):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if draw:
            letters_img = img.copy()
            contours_img = img.copy()
            cv2.drawContours(contours_img, contours, -1, (0, 255, 0), 2)
            cv2.imshow("contours", contours_img)
            cv2.waitKey(1)

        keys = {}

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if h / w > MAX_ASPECT or w / h > MAX_ASPECT or w * h < MIN_AREA:
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
                allowlist=ALLOWLIST + "ÆÅØ",
                detail=1,
            )

            if not result or len(result) != 1:
                continue

            text, conf = result[0][1], result[0][2]

            if draw and conf > 0.5:
                cv2.rectangle(letters_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.putText(
                    letters_img,
                    f"{text} ({conf:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

            if len(text) != 1 or conf < MIN_CONF or text not in ALLOWLIST:
                continue

            img_w = img.shape[1]
            if (text == "A" or text == "E") and x > img_w / 2:
                # discard A/E if they are mistaken for the Å/Æ keys on the right side of the keyboard
                continue

            if text in keys:
                # remove duplicate keys
                # only replace with none if saved conf is lower than 0.99 and current conf is lower than 0.99
                # if saved conf is higher than 0.99 and higher than current conf, keep saved
                # if current conf is higher than 0.99 and higher than saved conf, replace with current

                saved_conf = keys[text][0] or 0
                min_conf = 0.99

                if conf < min_conf and saved_conf < min_conf:
                    keys[text] = (None, (None, None), (None, None))
                    continue
                elif saved_conf > min_conf and saved_conf > conf:
                    continue
                # elif conf > min_conf and conf > saved_conf:
                #     pass

            keys[text] = (conf, (x, y), (w, h))

        detected_keys = {k: v for k, v in keys.items() if v[0] is not None}

        if draw:
            for k, (conf, (x, y), (w, h)) in detected_keys.items():
                cv2.rectangle(letters_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    letters_img,
                    f"{k} ({conf:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("letters", letters_img)
            cv2.waitKey(1)

        return detected_keys

    def keyboard_picture(self, out_dir="keyboard_images", flush=5):
        # Discard buffered frames so we get the image from NOW
        # (important after the robot moves)
        for _ in range(flush):
            self.cap.grab()
        raw = self.read()

        # Undistort, keeping the same K for the output image
        undist = cv2.undistort(raw, camera_matrix, dist, None, camera_matrix)

        # Save both
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = os.path.join(out_dir, f"keyboard_{stamp}_raw.png")
        undist_path = os.path.join(out_dir, f"keyboard_{stamp}_undist.png")
        cv2.imwrite(raw_path, raw)
        cv2.imwrite(undist_path, undist)
        print(f"Saved {raw_path} and {undist_path}")

        return undist, undist_path


if __name__ == "__main__":
    cam = Camera()
    try:
        while True:
            frame = cam.read()
            cv2.imshow("camera view", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("p"):  # p -> take keyboard picture
                img, path = cam.keyboard_picture()
                cv2.imshow("keyboard picture (undistorted)", img)
            elif key == ord("q"):
                break

    finally:
        cam.cap.release()
        cv2.destroyAllWindows()
