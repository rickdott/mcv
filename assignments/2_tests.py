from shared.Calibrator import Calibrator
import os
import numpy as np

calibrator = Calibrator('resources/img_*.png')

for cam in range(1, 5):
    # intrinsics_path = os.path.join('assignments', 'data', f'cam{cam}', 'config.npz')
    # calibrator.calibrate(False, False, savename=intrinsics_path)
    # print(calibrator.ret)
    # print(calibrator.mtx)
    config_path = os.path.join('assignments', 'data', f'cam{cam}', 'config.npz')
    with np.load(os.path.join(config_path)) as calibration:
        ret = calibration['ret']
        mtx = calibration['mtx']
        dist = calibration['dist']
        rvecs = calibration['rvecs']
        tvecs = calibration['tvecs']
        rvec = calibration['rvec']
        tvec = calibration['tvec']
    print(f'{cam}: INTRINSICS')
    print(ret)
    print(mtx)
    print(dist)
    print(f'{cam}: EXTRINSICS')
    print(rvec)
    print(tvec)
