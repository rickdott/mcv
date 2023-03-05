from shared.Calibrator import Calibrator
import os
import numpy as np
from shared.VoxelCam import BASE_PATH, VoxelCam
import cv2 as cv
import glm
from shared.VoxelReconstructor import RESOLUTION
from collections import defaultdict

print(all([True, True]))
print([[0,0,0]]*4)
# calibrator = Calibrator('resources/img_*.png')
# for cam in range(1, 5):
# #     # intrinsics_path = os.path.join('assignments', 'data', f'cam{cam}', 'config.npz')
# #     # calibrator.calibrate(False, False, savename=intrinsics_path)
# #     # print(calibrator.ret)
# #     # print(calibrator.mtx)
#     config_path = os.path.join('data', f'cam{cam}', 'config.npz')
#     with np.load(os.path.join(config_path)) as calibration:
#         ret = calibration['ret']
#         mtx = calibration['mtx']
#         dist = calibration['dist']
#         rvecs = calibration['rvecs']
#         tvecs = calibration['tvecs']
#         rvec = calibration['rvec']
#         tvec = calibration['tvec']
#     # # print(f'Cam{cam}')
#     # # print(mtx)
#     # # print(rvec.T * tvec)
#     rmtx = cv.Rodrigues(rvec)[0]
#     # pos = -rmtx.T * np.matrix(tvec)
#     # # print(pos)
#     print(f'Cam {cam} rotation matrix')
#     print(rmtx)
#     print(f'Cam {cam} translation vector')
#     print(tvec)
    # glm_m = glm.mat4(I)
    # rot_m = glm.rotate(glm_m, glm.radians(90), (0, 1, 0))
    # print(rot_m)
    # loc = rmtx.T * tvec
    # with np.load(os.path.join(os.path.join('assignments', 'data', f'cam{cam}', 'table.npz'))) as tbl:
    #     table = tbl['table']

    # print(list(set([1, 2, 3]) & set([3, 4, 5])))
    # table_d = {i:np.where(table==i)[0] for i in table}

#     # grid = np.float32(np.mgrid[-TABLE_SIZE[0]:TABLE_SIZE[0], -TABLE_SIZE[1]:TABLE_SIZE[1], 0:TABLE_SIZE[2]])
#     steps = 100j
#     grid = np.float32(np.mgrid[-TABLE_SIZE[0]:TABLE_SIZE[0]:steps, -TABLE_SIZE[1]:TABLE_SIZE[1]:steps, 0:TABLE_SIZE[2]])
#     # grid = np.float32(np.meshgrid(np.arange(0, TABLE_SIZE[0]), np.arange(0, TABLE_SIZE[1]), np.arange(0, TABLE_SIZE[2])))
#     grid_t = grid.T.reshape(-1, 3) * 115

#     # Project indices to 2d
#     proj_list = cv.projectPoints(grid_t, cam['rvec'], cam['tvec'], cam['mtx'], cam['dist'])[0]

#     # Create table of (X, Y, Z, 2) shape containing 2d coordinates
#     table = np.int_(proj_list).reshape(list(TABLE_SIZE) + [2], order='F')
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
