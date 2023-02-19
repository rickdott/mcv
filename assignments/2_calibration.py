import os
import cv2 as cv

from shared.Calibrator import Calibrator

# Obtain intrinsic camera parameters
# cv.VideoCapture('assignments/data/cam1/intrinsics.avi')
for cam in range(1, 4):
    intrinsics_path = os.path.join('assignments', 'data', f'cam{cam}', 'intrinsics.avi')
    vid = cv.VideoCapture(intrinsics_path)

    frames = []
    count = 0
    while vid.isOpened():
        ret, frame = vid.read()
        if ret:
            # cv.imshow('Intrinsics', frame)
            frames.add(frame)
            # Advance 250 frames == 5 seconds (video is 50fps)
            count += 250
            vid.set(cv.CAP_PROP_POS_FRAMES, count)

        else:
            vid.release()
            break
    
    
    calibrator = Calibrator()
    cv.waitKey(0)