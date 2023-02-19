import cv2 as cv
import numpy as np
from datetime import datetime

# Camera class that uses the webcam, used in the online phase of the assignment
class LiveCamera():

    def __init__(self, calibrator):
        # Constructor of LiveCamera class, incorporate existing calibrator
        self.calibrator = calibrator
        self.cube_size = self.calibrator.CELL_SIZE * 3

        # 8 points that make up the cube shape in coordinate (XYZ) space
        self.cube_points = np.float32([[0,0,0], [0,self.cube_size,0], [self.cube_size,self.cube_size,0], [self.cube_size,0,0],
                   [0,0,-self.cube_size], [0,self.cube_size,-self.cube_size], [self.cube_size,self.cube_size,-self.cube_size], [self.cube_size,0,-self.cube_size]])


    def start(self):
        # Start live webcam session, attempts to find chessboard corners,
        # then draw axes and 3d shape. Does this until the ESC button is pressed
        cam = cv.VideoCapture(0)
        cv.namedWindow("Live")

        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed")
                break

            # Convert to grayscale and attempt to find chessboardcorners 
            # use cv.CALIB_CB_FAST_CHECK since many webcam frames do not include chessboard
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, self.calibrator.board_size, None, flags=cv.CALIB_CB_FAST_CHECK)
            if ret:
                # Find better corner locations (more exact than integer pixels)
                corners_subpix = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), self.calibrator.CRITERIA)
                
                # Using subpixel corners and intrinsic camera matrix, find rotation and translation vectors for the current frame
                _, rvec, tvec = cv.solvePnP(self.calibrator.objp, corners_subpix, self.calibrator.mtx, self.calibrator.dist)

                # Draw frame axes (x, y, z) directions on frame
                cv.drawFrameAxes(frame, self.calibrator.mtx, self.calibrator.dist, rvec, tvec, self.calibrator.CELL_SIZE * (self.calibrator.board_size[1] - 1))

                # Find points in 2d image where cube_points should be given camera matrix and rvec/tvec
                projected_cube = cv.projectPoints(self.cube_points, rvec, tvec, self.calibrator.mtx, self.calibrator.dist)

                self.draw_cube(projected_cube, frame)

            # Show frame, even if checkersboard is not found
            cv.imshow("Live", frame)

            k = cv.waitKey(1)
            if k%256 == 27:
                # Escape pressed
                print("Quitting...")
                break
            elif k%256 == 32:
                # Spacebar
                print("Saving image...")
                now = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
                fname = f'img_{now}.png'
                cv.imwrite(fname, frame)
        
        # Clean up
        cv.destroyAllWindows()
        cam.release()

    def draw_cube(self, projected_cube, frame):
        # Intuition: From any corner, there should be a line to any other corner where only one coordinate changes
        amt_points = len(projected_cube[0])
        drawn_lines = []

        for point_i in range(amt_points):
            for point_j in range(amt_points):
                if point_i == point_j: continue

                # If sum of absolute differences between cube points is equal to cube_size
                if np.sum(np.abs(self.cube_points[point_i] - self.cube_points[point_j])) == self.cube_size:
                    point_x = tuple(projected_cube[0][point_i][0].astype(np.intc))
                    point_y = tuple(projected_cube[0][point_j][0].astype(np.intc))

                    # If line has not been drawn yet (checks both ways to avoid double lines)
                    if (point_x, point_y) not in drawn_lines and (point_y, point_x) not in drawn_lines:
                        # Draw line between points
                        cv.line(frame, point_x, point_y, (0, 255, 255), 2)
                        drawn_lines.append((point_x, point_y))