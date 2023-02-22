import os
import cv2 as cv
import numpy as np

from shared.Calibrator import Calibrator

for cam in range(1, 5):
    base_path = os.path.join('assignments', 'data', f'cam{cam}')
    extrinsics_path = os.path.join(base_path, 'checkerboard.avi')
    intrinsics_path = os.path.join(base_path, 'intrinsics.npz')

    vid = cv.VideoCapture(extrinsics_path)

    frames = []
    ret, frame = vid.read()
    if ret:
        frames.append(frame)
    else:
        vid.release()

    # Board and cell size adapted from checkerboard.xml
    calibrator = Calibrator(frames=frames, board_size=(8, 6), cell_size=115, intrinsics_path=intrinsics_path)
    calibrator.calibrate(recalibrate=True, save=True, savename=os.path.join('assignments', 'data', f'cam{cam}', 'config.npz'))

    # Added due to issues with lists inside calibrator remaining in new loop
    del calibrator
    vid.release()
    cv.destroyAllWindows()