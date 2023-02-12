import cv2 as cv

class LiveCamera():

    def __init__(self, calibrator):
        self.calibrator = calibrator

    def start(self):
        # Online phase
        cam = cv.VideoCapture(0)
        cv.namedWindow("Live")

        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed")
                break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, self.calibrator.BOARD_SIZE, None, flags=cv.CALIB_CB_FAST_CHECK)
            if ret:
                corners_subpix = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.calibrator.CRITERIA)
                _, rvec, tvec = cv.solvePnP(self.calibrator.objp, corners_subpix, self.calibrator.mtx, self.calibrator.dist)
                print(rvec)
                
                frame = cv.drawFrameAxes(frame, self.calibrator.mtx, self.calibrator.dist, rvec, tvec, 5)
            cv.imshow("Live", frame)

            k = cv.waitKey(1)
            if k%256 == 27:
                # Escape
                print("Quitting...")
                break

        cv.destroyAllWindows()
        cam.release()