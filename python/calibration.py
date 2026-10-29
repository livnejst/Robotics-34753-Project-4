import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# Implement the number of vertical and horizontal corners
nb_vertical = 6
nb_horizontal = 9


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nb_horizontal*nb_vertical,3), np.float32)
objp[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('calib_images/*.png')
assert images


for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(image = gray, patternSize=(nb_vertical, nb_horizontal))

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (nb_vertical,nb_horizontal), corners,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
img = cv2.imread('calib_images/calib_20261029_094456_838.png')
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

print(newcameramtx)
print(mtx)
print(dist)

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# original i undistorted una al costat de l'altra
both = np.hstack((img, dst))
cv2.putText(both, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(both, 'Undistorted', (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.namedWindow('Original vs undistorted', cv2.WINDOW_NORMAL)
cv2.imshow('Original vs undistorted', both)

# crop the image
x, y, w2, h2 = roi
dst_crop = dst[y:y+h2, x:x+w2]
cv2.namedWindow('Undistorted (crop)', cv2.WINDOW_NORMAL)
cv2.imshow('Undistorted (crop)', dst_crop)

cv2.waitKey(0)          # espera una tecla
cv2.destroyAllWindows()