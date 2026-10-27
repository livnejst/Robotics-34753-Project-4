import cv2

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

if __name__ == "__main__":
    cam = Camera()
    try:
        while True:
            frame = cam.read()
            cv2.imshow("camera view", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cam.cap.release()
        cv2.destroyAllWindows()