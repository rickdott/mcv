from Calibrator import Calibrator
import cv2 as cv
from LiveCamera import LiveCamera
# All files
# calibrator = Calibrator('resources/img*.png')
# Without manual files
calibrator = Calibrator('resources/img_*[!X].png')
calibrator.calibrate(recalibrate=False)

live = LiveCamera(calibrator)
live.start()




# # test_img = 'resources/img_31_BORDER.png'
# img = cv.imread(test_img)
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# ret, corners = cv.findChessboardCorners(gray, calibrator.BOARD_SIZE, None)

# if ret:
#     _, rvec, tvec, _ = cv.solvePnPRansac(calibrator.objp, corners, mtx, dist)
#     img = cv.drawFrameAxes(img, mtx, dist, rvec, tvec, 5)
    
#     cv.imshow('img', img)
#     cv.waitKey(0)



# img = cv.imread('resources/img_31_BORDER.png')
# h, w = img.shape[:2]
# newmtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# dst = cv.undistort(img, mtx, dist, None, newmtx)

# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imshow('img', dst)
# cv.waitKey(0)