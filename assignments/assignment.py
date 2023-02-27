import glm
import random
import numpy as np

from shared.VoxelReconstructor import VoxelReconstructor
import cv2 as cv
import multiprocessing as mp

block_size = 1.0


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data = []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
    return data


def set_voxel_positions(vc):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    data = vc.next_frame()
    return data




def get_cam_positions(vc):
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    positions = []
    for cam in vc.cams:
        # Calculate rotation matrix and determine position
        rmtx = cv.Rodrigues(cam.rvec)[0]
        pos = -rmtx.T * np.matrix(cam.tvec)

        # Swap Y and Z
        pos[[1, 2]] = pos[[2, 1]]

        # Take absolute of height variable (calibration had -z as positive height)
        pos[1] = abs(pos[1])
        # Divide by 115 to get cameras closer
        # TODO: Find better number after voxels show up
        positions.append(pos/115)
    return positions


def get_cam_rotation_matrices(vc):
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    rotations = []
    for cam in vc.cams:
        I = np.identity(4)
        rmtx = cv.Rodrigues(cam.rvec)[0]
        I[0:3,0:3] = rmtx
        
        glm_mat = glm.mat4(I)
        glm_mat = glm.rotate(glm_mat, glm.radians(90), (0, 1, 1))
        rotations.append(glm_mat)
    return rotations
