import numpy as np
import cv2 as cv
import glob
import os
from operator import itemgetter

# Calibrator processes the images meant for calibration and calibrates the camera
class Calibrator:

    CELL_SIZE = 24#mm

    CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    EPSILON = 0.01

    objpoints = []
    imgpoints = []

    def __init__(self, path=None, frames=None, board_size=(9,6)):
        # Constructor of Calibrator class, finds paths of images that should be used
        self.path = path
        if path is not None:
            self.images = glob.glob(path)
        if frames is not None:
            self.frames = frames
        self.click_coords = []
        self.manual_corners = []
        self.prev_ret = 999

        self.board_size = board_size
        self.objp = np.zeros((self.board_size[1]*self.board_size[0], 3), np.float32)
        # Adjust object point size for cell size, spaces each square by CELL_SIZE
        self.objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2) * self.CELL_SIZE

    def interpolate(self, start_coord, end_coord, stepsize):
        # Linearly interpolate between two coordinates, finding stepsize
        # amount of equidistant points between start_coord and end_coord
        x = np.linspace(start_coord[0], end_coord[0], stepsize)
        y = np.linspace(start_coord[1], end_coord[1], stepsize)

        return list(zip(x, y))

    def click_event(self, event, x, y, flags, param):
        # Event triggered on click during manual annotation of checkersboard corners
        if event == cv.EVENT_LBUTTONDOWN:
            print(f'Clicked at coordinate ({x},{y})')
            self.click_coords.append((x, y))
            cv.circle(param['img'], (x, y), 2, (0, 0, 255), thickness=cv.FILLED)
            cv.imshow('img', param['img'])

            if len(self.click_coords) == 4:
                # Looks within masked area (between 4 points) for corners up to a maximum of board_x*board_y
                mask = np.zeros(self.gray.shape, dtype=np.uint8)
                cv.fillConvexPoly(mask, np.expand_dims(np.array(self.click_coords, dtype=np.int32), 1), 1)
                corners = cv.goodFeaturesToTrack(image=self.gray, maxCorners=self.board_size[0] * self.board_size[1], qualityLevel=0.01, minDistance=int(self.CELL_SIZE * 0.8), corners=None, mask=mask, blockSize=3, gradientSize=3, useHarrisDetector=False, k=0.04)

                if corners.shape[0] < self.board_size[0] * self.board_size[1]:
                    corners = []
                    print('Not enough corners found automatically, falling back to linear interpolation')
                    # Not enough corners were detected, fall back to linear interpolation
                    left_vert = self.interpolate(self.click_coords[0], self.click_coords[3], self.board_size[1])
                    right_vert = self.interpolate(self.click_coords[1], self.click_coords[2], self.board_size[1])

                    for left, right in zip(left_vert, right_vert):
                        corners.extend(self.interpolate(left, right, self.board_size[0]))

                    corners = np.array(corners, dtype=np.float32)
                    corners = np.expand_dims(corners, 1)

                corners_subpix = cv.cornerSubPix(self.gray, corners, (13, 13), (-1, -1), self.CRITERIA)
                for i in range(corners_subpix.shape[0]):
                    cv.circle(param['img'], (int(corners_subpix[i,0,0]), int(corners_subpix[i,0,1])), 2, (0, 255, 0), cv.FILLED)

                cv.imshow('img', param['img'])
                self.click_coords = []
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners_subpix)
                cv.imshow('img', param['img'])

    def process_images(self):
        if self.path is not None:
            for idx, fname in enumerate(self.images):
                frame = cv.imread(fname)
                self.process_image(frame, idx)
        else:
            for idx, frame in enumerate(self.frames):
                self.process_image(frame, idx)
        cv.destroyAllWindows()

    def process_image(self, frame, index):
        # Processes all images at given location, automatically or manually detecting chessboard corners
        self.gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(self.gray, self.board_size, None)

        if ret:
            # Corners are found
            self.objpoints.append(self.objp)
            corners_subpix = cv.cornerSubPix(self.gray, corners, (11, 11), (-1, -1), self.CRITERIA)
            self.imgpoints.append(corners_subpix)

            cv.drawChessboardCorners(frame, self.board_size, corners_subpix, ret)
            cv.imshow('img', frame)
            cv.waitKey(5)
        else:
            # Corners not found, manually request corners
            print(f'Did not find corners for frame: {index}')

            cv.imshow('img', frame)
            cv.setMouseCallback('img', self.click_event, param={'img': frame})
            cv.waitKey(0)
        
        # Calibrate camera and only use new image if projection error does not decrease by more than epsilon
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints, self.gray.shape[::-1], None, None)
        if self.ret - self.prev_ret > self.EPSILON:
            print(f'Frame {index} made projection error worse by: {self.ret - self.prev_ret:.4}')
            self.objpoints.pop()
            self.imgpoints.pop()
        self.prev_ret = self.ret
    
    def calibrate(self, recalibrate=False, save=True, savename='calibration.npz'):
        if not recalibrate:
            with np.load(os.path.join(savename)) as calibration:
                self.ret = calibration['ret']
                self.mtx = calibration['mtx']
                self.dist = calibration['dist']
                self.rvecs = calibration['rvecs']
                self.tvecs = calibration['tvecs']
        else:
            self.process_images()
            # Final calibration with only improving frames
            self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints, self.gray.shape[::-1], None, None)
            print(f"Re-projection error: {self.ret:.4}")
            if save:
                np.savez(os.path.join(savename), ret=self.ret, mtx=self.mtx, dist=self.dist, rvecs=self.rvecs, tvecs=self.tvecs)
