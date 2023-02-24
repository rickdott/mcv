import os
from shared.VoxelCam import VoxelCam
import numpy as np
import cv2 as cv
import multiprocessing as mp

TABLE_SIZE = (128, 128, 64)

# Create 3d to 2d lookup table
# use cv.projectPoints to project all points in needed (X, Y, Z) space to (X, Y) space for each camera
def calc_table(cam):
    table = np.empty(TABLE_SIZE, dtype=tuple)
    for x in range(table.shape[0]):
        for y in range(table.shape[1]):
            for z in range(table.shape[2]):
                table[x, y, z] = tuple(cv.projectPoints(np.float32([x, y, z]), cam['rvec'], cam['tvec'], cam['mtx'], cam['dist'])[0][0][0])
    return (cam['idx'], table)

def is_in_all_foregrounds(coords, foregrounds):
    x, y, z = coords
    for fg in foregrounds:
        if fg[x, y, z] == 0:
            return False
    return True

if __name__ == "__main__":
    cams = []
    cam_infos = []

    # Create VoxelCam instances and pre-load their pickle-able information sets
    for cam in range(1, 5):
        vcam = VoxelCam(cam)
        cams.append(vcam)
        cam_infos.append(vcam.get_info())
    
    # Parallelized calculation of lookup table
    with mp.Pool(len(cams)) as p:
        results = p.map(calc_table, cam_infos)
    
    reconstruction = np.ones(TABLE_SIZE)

    foregrounds = []
    while True:
        for cam in cams:
            cam.next_frame()
            foregrounds.append(cam.get_foreground())
        for x in range(reconstruction.shape[0]):
            for y in range(reconstruction.shape[1]):
                for z in range(reconstruction.shape[2]):
                    if not is_in_all_foregrounds((x, y, z), foregrounds):
                        reconstruction[x, y, z] = 0


                            



    # frames = []
    # for cam in cams:
    #     frames.append(cam.next_frame())
    # for frame in video
    #     remove background