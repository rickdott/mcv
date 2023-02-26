from shared.Calibrator import Calibrator
import os
import numpy as np
from shared.VoxelCam import BASE_PATH, VoxelCam
import cv2 as cv

# calibrator = Calibrator('resources/img_*.png')

# for cam in range(1, 5):
#     # intrinsics_path = os.path.join('assignments', 'data', f'cam{cam}', 'config.npz')
#     # calibrator.calibrate(False, False, savename=intrinsics_path)
#     # print(calibrator.ret)
#     # print(calibrator.mtx)
#     config_path = os.path.join('assignments', 'data', f'cam{cam}', 'config.npz')
#     with np.load(os.path.join(config_path)) as calibration:
#         ret = calibration['ret']
#         mtx = calibration['mtx']
#         dist = calibration['dist']
#         rvecs = calibration['rvecs']
#         tvecs = calibration['tvecs']
#         rvec = calibration['rvec']
#         tvec = calibration['tvec']
#     print(f'{cam}: INTRINSICS')
#     print(ret)
#     print(mtx)
#     print(dist)
#     print(f'{cam}: EXTRINSICS')
#     print(rvec)
#     print(tvec)

# with np.load(os.path.join(BASE_PATH, f'cam1', 'table.npz'), allow_pickle=True) as f_table:
#     print(type(f_table['table']))

# cams = [1, 2, 3, 4]
# fgs = [5, 6, 7, 8]

# for c, f in zip(cams, fgs):
#     print(c, f)
# with np.load(os.path.join(BASE_PATH, f'cam1', 'config.npz')) as calibration:
#     mtx = calibration['mtx']
#     dist = calibration['dist']
#     rvec = calibration['rvec']
#     tvec = calibration['tvec']
# test = np.ones((10, 12, 14))
# shap = (10, 12, 14)
# grid = np.float32(np.meshgrid(np.arange(0, 10), np.arange(0, 12), np.arange(0, 14))).T.reshape(-1, 3) * 112
# proj_inds = cv.projectPoints(grid, rvec, tvec, mtx, dist)

# proj_arr = np.array(proj_inds[0]).reshape(tuple(list(shap) + [2]))
# # proj_arr[:, :, :, 0:1] = tuple(proj_arr[:,:,:, 0:1])
# # proj_arr[:, 0, :] = tuple(proj_arr[:, 0, :])
# print(proj_inds[0])

# print((list((12, 12)), 6))

# print(np.ones([10, 10]))

# 
for i in range(int(-10/2), int(10/2)):
    print(i)