from shared.Calibrator import Calibrator

calibrator = Calibrator('resources/img_*.png')

calibrator.calibrate(False, False, savename='calibration_run1_2.npz')
print(calibrator.ret)
print(calibrator.mtx)
calibrator.calibrate(False, False, savename='calibration_run2_2.npz')
print(calibrator.ret)
print(calibrator.mtx)
calibrator.calibrate(False, False, savename='calibration_run3_2.npz')
print(calibrator.ret)
print(calibrator.mtx)
