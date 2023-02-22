import numpy as np
import cv2 as cv
import glob
import os
from operator import itemgetter

# Calibrator processes the images meant for calibration and calibrates the camera
class Calibrator:

    CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    EPSILON = 0.01

    objpoints = []
    imgpoints = []

    def __init__(self, path=None, frames=None, board_size=(9,6), cell_size=24, intrinsics_path=None):
        # Constructor of Calibrator class, finds paths of images that should be used
        self.path = path
        if path is not None:
            self.images = glob.glob(path)
        if frames is not None:
            self.frames = frames
        if intrinsics_path is not None:
            self.intrinsics_path = intrinsics_path
            with np.load(os.path.join(intrinsics_path)) as calibration:
                self.ret = calibration['ret']
                self.mtx = calibration['mtx']
                self.dist = calibration['dist']
                self.rvecs = calibration['rvecs']
                self.tvecs = calibration['tvecs']
        self.click_coords = []
        self.manual_corners = []
        self.prev_ret = 1
        self.cell_size = cell_size

        self.board_size = board_size
        self.objp = np.zeros((self.board_size[1]*self.board_size[0], 3), np.float32)
        # Adjust object point size for cell size, spaces each square by cell_size
        self.objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2) * self.cell_size

    def interpolate(self, start_coord, end_coord, stepsize):
        # Linearly interpolate between two coordinates, finding stepsize
        # amount of equidistant points between start_coord and end_coord
        x = np.linspace(start_coord[0], end_coord[0], stepsize)
        y = np.linspace(start_coord[1], end_coord[1], stepsize)

        return list(zip(x, y))

    def four_point_transform(self):
        # obtain a consistent order of the points and unpack them
        # individually
        points = np.array(self.click_coords, dtype=np.float32)

        (tl, tr, br, bl) = points
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        # compute the perspective transform matrix and then apply it
        M = cv.getPerspectiveTransform(points, dst)
        warped = cv.warpPerspective(self.frame, M, (maxWidth, maxHeight))
        # return the warped image
        return warped, M

    def click_event(self, event, x, y, flags, param):
        # Event triggered on click during manual annotation of checkersboard corners
        if event == cv.EVENT_LBUTTONDOWN:
            print(f'Clicked at coordinate ({x},{y})')
            self.click_coords.append((x, y))
            cv.circle(param['img'], (x, y), 2, (0, 0, 255), thickness=cv.FILLED)
            cv.imshow('img', param['img'])

            if len(self.click_coords) == 4:
                if self.intrinsics_path is not None:
                    # Calculate extrinsics only, perspective transform
                    warped, M = self.four_point_transform()
                    cv.imshow('warp', warped)
                    warped_gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
                    left_vert = self.interpolate((0, 0), (0, warped.shape[0]), self.board_size[1])
                    right_vert = self.interpolate((warped.shape[1], 0), (warped.shape[1], warped.shape[0]), self.board_size[1])
                    corners = []
                    for left, right in zip(left_vert, right_vert):
                        corners.extend(self.interpolate(left, right, self.board_size[0]))

                    corners = np.array(corners, dtype=np.float32)
                    corners = np.expand_dims(corners, 1)
                    corners_subpix = cv.cornerSubPix(warped_gray, corners, (5, 5), (-1, -1), self.CRITERIA)
                    
                    for i in range(corners_subpix.shape[0]):
                        cv.circle(warped, (int(corners_subpix[i,0,0]), int(corners_subpix[i,0,1])), 1, (0, 255, 0), cv.FILLED)
                    cv.imshow('warp', warped)


                    _, IM = cv.invert(M)
                    for i in range(corners_subpix.shape[0]):
                        x = corners_subpix[i, 0, 0]
                        y = corners_subpix[i, 0, 1]
                        
                        point = np.float32([x, y] + [1])
                        x, y, z = np.dot(IM, point)
                        new_x = x/z
                        new_y = y/z
                        corners_subpix[i, 0, 0] = new_x
                        corners_subpix[i, 0, 1] = new_y
                        cv.circle(self.frame, (int(new_x), int(new_y)), 1, (0, 255, 0), 1, cv.FILLED)
                    cv.imshow('img', self.frame)
                    self.imgpoints.append(corners_subpix)

                    # self.imgpoints.append(corners_subpix_dewarped)
                    # for corner in corners:
                    #     cv.circle(warped, tuple(corner[0].astype(np.uint32)), 1, (0, 0, 255), 1, cv.FILLED)

                else:
                    # Looks within masked area (between 4 points) for corners up to a maximum of board_x*board_y
                    mask = np.zeros(self.gray.shape, dtype=np.uint8)
                    cv.fillConvexPoly(mask, np.expand_dims(np.array(self.click_coords, dtype=np.int32), 1), 1)
                    corners = cv.goodFeaturesToTrack(image=self.gray, maxCorners=self.board_size[0] * self.board_size[1], qualityLevel=0.01, minDistance=int(self.cell_size * 0.8), corners=None, mask=mask, blockSize=3, gradientSize=3, useHarrisDetector=False, k=0.04)

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
                    self.imgpoints.append(corners_subpix)
                    cv.imshow('img', param['img'])

                self.click_coords = []
                self.objpoints.append(self.objp)

    def process_images(self):
        # Processes all images at given location, automatically or manually detecting chessboard corners, supports filenames for images and a list of frames for video
        if self.path is not None:
            for idx, fname in enumerate(self.images):
                frame = cv.imread(fname)
                self.process_image(frame, idx)
        else:
            for idx, frame in enumerate(self.frames):
                self.process_image(frame, idx)
        cv.destroyAllWindows()

    def process_image(self, frame, index):
        # Process a single frame, finding chessboard corners and calibrating the camera
        self.frame = frame
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
        
        # Dont use reprojection error to recalibrate if intrinsics is given
        if self.intrinsics_path is not None:
            return

        # Calibrate camera and only use new image if projection error does not decrease by more than epsilon
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints, self.gray.shape[::-1], None, None)
        print(f"Re-projection error: {self.ret:.4}")
        if self.ret - self.prev_ret > self.EPSILON:
            print(f'Frame {index} made projection error worse by: {self.ret - self.prev_ret:.4}')
            self.objpoints.pop()
            self.imgpoints.pop()
        self.prev_ret = self.ret
    
    def calibrate(self, recalibrate=False, save=True, savename='calibration.npz'):
        # Loads existing calibration or calibrates the camera
        if not recalibrate:
            with np.load(os.path.join(savename)) as calibration:
                self.ret = calibration['ret']
                self.mtx = calibration['mtx']
                self.dist = calibration['dist']
                self.rvecs = calibration['rvecs']
                self.tvecs = calibration['tvecs']
        else:
            self.process_images()
            if self.intrinsics_path is None:
                # Final calibration with only improving frames
                self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints, self.gray.shape[::-1], None, None)
                if save:
                    np.savez(os.path.join(savename), ret=self.ret, mtx=self.mtx, dist=self.dist, rvecs=self.rvecs, tvecs=self.tvecs)
            else:
                # Use objp and first entry in imgpoints (all corners found in the frame) to calibrate extrinsics
                _, rvec, tvec = cv.solvePnP(self.objp, self.imgpoints[0], self.mtx, self.dist)
                # Use solvePnP to find extrinsics for img/cam
                # Draw frame axes (x, y, z) directions on frame
                cv.drawFrameAxes(self.frame, self.mtx, self.dist, rvec, tvec, self.cell_size * (self.board_size[1] - 1), thickness=1)
                cv.imshow('img', self.frame)
                cv.waitKey(0)
                rmtx = cv.Rodrigues(rvec)
                print('Hi!')