import cv2 as cv
import numpy as np

# Camera class that uses the webcam, used in the online phase of the assignment
class LiveCamera():

    def __init__(self, calibrator):
        # Constructor of LiveCamera class, incorporate existing calibrator
        self.calibrator = calibrator
        cube_size = self.calibrator.CELL_SIZE * 3
        self.cube_points = np.float32([[0,0,0], [0,cube_size,0], [cube_size,cube_size,0], [cube_size,0,0],
                   [0,0,-cube_size],[0,cube_size,-cube_size],[cube_size,cube_size,-cube_size],[cube_size,0,-cube_size] ])

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
            # use cv.CALIB_CB_FAST_CHECK since many webcam frames do not include checkersboard
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, self.calibrator.BOARD_SIZE, None, flags=cv.CALIB_CB_FAST_CHECK)
            if ret:
                # Find better corner locations (more exact than integer pixels)
                corners_subpix = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.calibrator.CRITERIA)
                
                # Using subpixel corners and intrinsic camera matrix, find rotation and translation vectors for the current frame
                _, rvec, tvec = cv.solvePnP(self.calibrator.objp, corners_subpix, self.calibrator.mtx, self.calibrator.dist)

                # Draw frame axes (x, y, z) directions on frame
                frame = cv.drawFrameAxes(frame, self.calibrator.mtx, self.calibrator.dist, rvec, tvec, self.calibrator.CELL_SIZE * (self.calibrator.BOARD_SIZE[1] - 1))

                # Draw cube
                # cube_points = np.array([1, 1, 1], dtype=np.float32)

                projected_cube = cv.projectPoints(self.cube_points, rvec, tvec, self.calibrator.mtx, self.calibrator.dist)

                amt_points = len(projected_cube[0])
                for point in range(amt_points):
                    if point < amt_points - 1:
                        cv.line(frame,(int(projected_cube[0][point][0][0]), int(projected_cube[0][point][0][1])), (int(projected_cube[0][point + 1][0][0]), int(projected_cube[0][point + 1][0][1])), (0, 255, 0), 2)

                # point_iterator = iter(projected_cube[0])
                
                # for point in point_iterator:
                #     next_point = next(point_iterator)
                #     cv.line(frame, (int(point[0][0]), int(point[0][1])), (int(next_point[0][0]), int(next_point[0][1])), (0, 255, 0), 2)
                for point in projected_cube[0]:
                    cv.circle(frame, (int(point[0][0]), int(point[0][1])), 2, (0, 255, 0), thickness=cv.FILLED)\
                
            # Show frame, even if checkersboard is not found
            cv.imshow("Live", frame)

            k = cv.waitKey(1)
            if k%256 == 27:
                # Escape pressed
                print("Quitting...")
                break
        
        # Clean up
        cv.destroyAllWindows()
        cam.release()