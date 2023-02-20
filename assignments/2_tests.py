from shared.Calibrator import Calibrator
import os

calibrator = Calibrator('resources/img_*.png')

for cam in range(1, 5):
    intrinsics_path = os.path.join('assignments', 'data', f'cam{cam}', 'intrinsics.npz')
    calibrator.calibrate(False, False, savename=intrinsics_path)
    print(calibrator.ret)
    print(calibrator.mtx)