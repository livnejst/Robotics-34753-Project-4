import cv2
import numpy as np
import os
from datetime import datetime

camera_matrix = np.array([[676.65668327,   0.        , 290.95785852],
                          [  0.        , 675.22254615, 232.87089645],
                          [  0.        ,   0.        ,   1.        ]])

dist = np.array([[-0.22241378, 1.62268073, 0.00668074, -0.02273341, -3.49198783]])

new_camera_matrix = np.array([[641.85616876,   0.        , 271.69745724],
                          [  0.        , 650.92799605, 235.33183707],
                          [  0.        ,   0.        ,   1.        ]])

class Camera:
    def __init__(self):
        self.index = 2      #Index of the robot camera in my PC
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("CAMERA COULD NOT OPEN")

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("COULD NOT READ FRAME")
        return frame        

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
            if key == ord("p"):                       # p -> take keyboard picture
                img, path = cam.keyboard_picture()
                cv2.imshow("keyboard picture (undistorted)", img)
            elif key == ord("q"):
                break

    finally:
        cam.cap.release()
        cv2.destroyAllWindows()