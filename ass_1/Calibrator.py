import numpy as np
import cv2 as cv
import glob
from operator import itemgetter

class Calibrator:

    CELL_SIZE = 24#mm
    BOARD_SIZE = (6, 9)

    CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((9*6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    def __init__(self, path):
        self.path = path
        self.images = glob.glob(path)
        self.click_coords = []
        self.manual_corners = []

    def interpolate(self, start_coord, end_coord, stepsize):
        x = np.linspace(start_coord[0], end_coord[0], stepsize)
        y = np.linspace(start_coord[1], end_coord[1], stepsize)

        return list(zip(x, y))

    def click_event(self, event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            print(f'Clicked at coordinate ({x},{y})')
            self.click_coords.append((x, y))

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
                self.manual_corners = np.array(self.manual_corners, dtype=float)
                self.manual_corners = np.expand_dims(self.manual_corners, 1)
                self.objpoints.append(self.objp)
                self.imgpoints.append(self.manual_corners)
                

    def calibrate(self):
        for fname in self.images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            ret, corners = cv.findChessboardCorners(gray, self.BOARD_SIZE, None)

            if ret:
                self.objpoints.append(self.objp)
                corners_subpix = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.CRITERIA)
                self.imgpoints.append(corners_subpix)
                cv.drawChessboardCorners(img, self.BOARD_SIZE, corners_subpix, ret)
                cv.imshow('img', img)
                cv.waitKey(500)
            else:
                print(f'Did not find corners for file: {fname}')
                # Manually request corners
                cv.imshow('img', img)
                cv.setMouseCallback('img', self.click_event)
                cv.waitKey(0)

                for point in self.manual_corners:
                    cv.circle(img, (int(point[0][0]), int(point[0][1])), 2, (0, 255, 0))
                cv.imshow('img', img)
                cv.waitKey(500)


        cv.destroyAllWindows()