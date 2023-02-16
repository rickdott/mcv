import numpy as np
import cv2 as cv
import glob
from operator import itemgetter

# Calibrator processes the images meant for calibration and calibrates the camera
class Calibrator:

    CELL_SIZE = 24#mm
    BOARD_SIZE = (9, 6)
    CALIBRATION_PATH = 'resources/calibration.npz'
    CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((BOARD_SIZE[1]*BOARD_SIZE[0], 3), np.float32)
    # Adjust object point size for cell size, spaces each square by CELL_SIZE
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2) * CELL_SIZE

    objpoints = []
    imgpoints = []

    def __init__(self, path):
        # Constructor of Calibrator class, finds paths of images that should be used
        self.path = path
        self.images = glob.glob(path)
        self.click_coords = []
        self.manual_corners = []

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
                
                print(self.click_coords)
                # Determine top left and bottom right coordinates
                # Top left is smallest sum of x+y
                top_left = min(self.click_coords)
                self.click_coords.remove(top_left)
                
                # Bottom right is biggest sum of x+y
                bottom_right = max(self.click_coords)
                self.click_coords.remove(bottom_right)

                # Top right is the one with the biggest x coordinate 
                # out of the remaining two pairs
                top_right = max(self.click_coords, key=itemgetter(0))
                self.click_coords.remove(top_right)

                # Bottom left is the only remaining coordinate
                bottom_left = self.click_coords[0]

                # Empty list for next use
                self.click_coords.remove(bottom_left)

                # Interpolate
                left_vert = self.interpolate(top_left, bottom_left, self.BOARD_SIZE[0])
                right_vert = self.interpolate(top_right, bottom_right, self.BOARD_SIZE[0])
                
                for left, right in zip(left_vert, right_vert):
                    self.manual_corners.extend(self.interpolate(left, right, self.BOARD_SIZE[1]))
                self.manual_corners = np.array(self.manual_corners, dtype=np.float32)
                self.manual_corners = np.expand_dims(self.manual_corners, 1)

                # Add objp and manually found corners to designated lists
                self.objpoints.append(self.objp)
                self.imgpoints.append(self.manual_corners)

                for point in self.manual_corners:
                    cv.circle(param['img'], (int(point[0][0]), int(point[0][1])), 2, (0, 255, 0), thickness=cv.FILLED)
                self.manual_corners = []
                cv.imshow('img', param['img'])

                
    def process_images(self):
        for fname in self.images:
            img = cv.imread(fname)
            self.gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            ret, corners = cv.findChessboardCorners(self.gray, self.BOARD_SIZE, None)

            if ret:
                self.objpoints.append(self.objp)
                corners_subpix = cv.cornerSubPix(self.gray, corners, (11, 11), (-1, -1), self.CRITERIA)
                self.imgpoints.append(corners_subpix)
                cv.drawChessboardCorners(img, self.BOARD_SIZE, corners_subpix, ret)
                cv.imshow('img', img)
                cv.waitKey(5)
            else:
                print(f'Did not find corners for file: {fname}')
                # Manually request corners
                cv.imshow('img', img)
                cv.setMouseCallback('img', self.click_event, param={'img': img})
                cv.waitKey(0)

        cv.destroyAllWindows()
    
    def calibrate(self, recalibrate=False, save=True):
        if not recalibrate:
            with np.load(self.CALIBRATION_PATH) as calibration:
                self.ret = calibration['ret']
                self.mtx = calibration['mtx']
                self.dist = calibration['dist']
                self.rvecs = calibration['rvecs']
                self.tvecs = calibration['tvecs']
        else:
            self.process_images()
            self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints, self.gray.shape[::-1], None, None)
            if save:
                np.savez(self.CALIBRATION_PATH, ret=self.ret, mtx=self.mtx, dist=self.dist, rvecs=self.rvecs, tvecs=self.tvecs)
