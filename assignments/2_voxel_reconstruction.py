import os
from shared.VoxelCam import VoxelCam
import numpy as np
import cv2 as cv

# Timing
import time
import multiprocessing as mp


cams = []
# Load all 5 videos
# Load calibration matrices
# Load all 5 background subtractors
for cam in range(1, 5):
    cams.append(VoxelCam(cam))

# Create 3d to 2d lookup table
# use cv.projectPoints to project all points in needed (X, Y, Z) space to (X, Y) space for each camera
# Parallelize on each cam? Otherwise no feasible time gain
# start_time = time.time()
# for cam in cams:
#     print(f'Filling table for cam {cam.idx}')
#     for x in range(cam.table.shape[0]):
#         for y in range(cam.table.shape[1]):
#             for z in range(cam.table.shape[2]):
#                 cam.table[x, y, z] = cv.projectPoints(np.float32([x, y, z]), cam.rvec, cam.tvec, cam.mtx, cam.dist)
# print(time.time() - start_time)
def calc_table(cam):
    for x in range(cam['table'].shape[0]):
        for y in range(cam['table'].shape[1]):
            for z in range(cam['table'].shape[2]):
                cam['table'][x, y, z] = cv.projectPoints(np.float32([x, y, z]), cam['rvec'], cam['tvec'], cam['mtx'], cam['dist'])

if __name__ == "__main__":
    start_time = time.time()
    threads = {}
    mp.set_start_method('spawn')
    for cam in cams:
        print(f'Filling table for cam {cam.idx}')
        threads[cam.idx] = mp.Process(target=calc_table, args=(cam.get_info(),))

    threads[1].start()
    threads[2].start()
    threads[3].start()
    threads[4].start()

    threads[1].join()
    threads[2].join()
    threads[3].join()
    threads[4].join()

    print(time.time() - start_time)

    # frames = []
    # for cam in cams:
    #     frames.append(cam.next_frame())
    # for frame in video
        # remove background