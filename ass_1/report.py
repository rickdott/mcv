from Calibrator import Calibrator

calibrator = Calibrator('resources/img_*[!X].png')
ret, mtx, dist, rvecs, tvecs = calibrator.calibrate(recalibrate=False, save=False)
