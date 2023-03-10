import numpy as np, cv2 as cv
from shared.VoxelCam import VoxelCam, BASE_PATH
import multiprocessing as mp
import os
from collections import defaultdict

# Resolution to use when generating the voxel model
RESOLUTION = 50

# The VoxelReconstruct class handles calculating  the lookup tables for individual VoxelCams
# and gathering the voxel, color combinations for the next frame.
class VoxelReconstructor():

    def __init__(self, create_table=False):
        # Create VoxelCam instances and pre-load their pickle-able information sets
        self.cams = []
        self.cam_infos = []

        for cam in range(1, 5):
            vcam = VoxelCam(cam)
            self.cams.append(vcam)
            self.cam_infos.append(vcam.get_info())
        
        self.cam_amount = len(self.cams)

        # (X, Y, Z) -> count, color dicts
        self.voxels = defaultdict(lambda: [False] * self.cam_amount)
        self.colors = defaultdict(lambda: [[0, 0, 0]] * self.cam_amount)

        if create_table:
            # Parallelized calculation of lookup table
            with mp.Pool(len(self.cams)) as p:
                results = p.map(calc_table, self.cam_infos)
            
            for result in results:
                self.cams[result[0] - 1].table = result[1]

            # Save tables to disk
            for idx, cam in enumerate(self.cams):
                table_path = os.path.join(BASE_PATH, f'cam{idx + 1}', 'table.npz')
                np.savez(table_path, table=cam.table)
        else:
            for idx, cam in enumerate(self.cams):
                table_path = os.path.join(BASE_PATH, f'cam{idx + 1}', 'table.npz')
                with np.load(table_path, allow_pickle=True) as f_table:
                    cam.table = f_table['table']

    # Selects the changed voxels + colors for the next frame
    def next_frame(self):
        next_voxels = []
        next_colors = []

        # For every cam/view, advance it one frame and receive the changed pixels
        for cam in self.cams:
            ret = cam.next_frame()
            if not ret:
                return
            changed = cam.xor.nonzero()

            print(f'{cam.idx} Changed pixels: {len(changed[0])}')

            # For every changed foreground pixel, set it's show-value (True/False) for the cam it is shown on
            # also add the color of the foreground pixel for the current camera
            for pix_y, pix_x in zip(changed[0], changed[1]):
                coord = (pix_x, pix_y)
                for voxel in cam.table[coord]:
                    # Set show-value of voxel based on foreground value of changed pixel
                    self.voxels[voxel][cam.idx-1] = cam.fg[pix_y, pix_x] != 0

                    if cam.fg[pix_y, pix_x] != 0:
                        self.colors[voxel][cam.idx-1] = cam.frame[pix_y, pix_x]

        # For all the voxel, color combinations, add those that occurred in all cameras
        for voxel, check in self.voxels.items():
            if all(check):
                # Add voxel including offsetting due to calibration and to place it in the middle of the plane
                next_voxels.append([voxel[0]-(int(RESOLUTION)/2), -voxel[2]+RESOLUTION, voxel[1]-(int(RESOLUTION)/2)])

                # Divide by 100 to get color in [0, 1] interval
                color = np.mean(np.array(self.colors[voxel]), axis=0) / 100

                # BGR to RGB (for OpenGL)
                color[[0, 2]] = color[[2, 0]]
                
                next_colors.append(color)

        return next_voxels, next_colors

# Create (X, Y) -> [(X, Y, Z), ...] dictionary
# use cv.projectPoints to project all points in needed (X, Y, Z) space to (X, Y) space for each camera
def calc_table(cam):
    print(f'{cam["idx"]} Calculating table')
    steps_c = complex(RESOLUTION)
    # Create evenly spaced grid centered around the subject, using n = RESOLUTION steps.
    grid = np.float32(np.mgrid[-400:1100:steps_c, -750:750:steps_c, -1500:0:steps_c])
    grid_t = grid.T.reshape(-1, 3)

    print(f'{cam["idx"]} Projecting points')
    # Project 3d voxel locations to 2d
    proj_list = cv.projectPoints(grid_t, cam['rvec'], cam['tvec'], cam['mtx'], cam['dist'])[0]

    # Create table of (X, Y, Z, 2) shape containing 2d coordinates
    table = np.int_(proj_list).reshape((RESOLUTION, RESOLUTION, RESOLUTION, 2), order='F')

    # Create dictionary: (X,Y) -> [(X, Y, Z), (X, Y,Z), ...]
    table_d = defaultdict(list)
    for x in range(RESOLUTION):
        for y in range(RESOLUTION):
            for z in range(RESOLUTION):
                table_d[tuple(table[x,y,z,:])].append((x,y,z))

    print(f'{cam["idx"]} Wrapping up')
    return (cam['idx'], table_d)