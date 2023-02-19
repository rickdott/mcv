from shared.Calibrator import Calibrator

calibrator = Calibrator('resources/img_*.png')

calibrator.calibrate(False, False, savename='resources/calibration_run1.npz')
print(calibrator.ret)
print(calibrator.mtx)
calibrator.calibrate(False, False, savename='resources/calibration_run2.npz')
print(calibrator.ret)
print(calibrator.mtx)
calibrator.calibrate(False, False, savename='resources/calibration_run3.npz')
print(calibrator.ret)
print(calibrator.mtx)
