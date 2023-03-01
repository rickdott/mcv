import numpy as np, cv2 as cv
from shared.VoxelCam import VoxelCam, BASE_PATH
import multiprocessing as mp
import os
from collections import defaultdict
from itertools import permutations

RESOLUTION = 100

class VoxelReconstructor():

    def __init__(self, create_table=False):
        # Create VoxelCam instances and pre-load their pickle-able information sets
        self.cams = []
        self.cam_infos = []
        for cam in range(1, 5):
            vcam = VoxelCam(cam)
            self.cams.append(vcam)
            self.cam_infos.append(vcam.get_info())

        if create_table:
            # Parallelized calculation of lookup table
            with mp.Pool(len(self.cams)) as p:
                results = p.map(calc_table, self.cam_infos)
            
            for result in results:
                self.cams[result[0] - 1].table = result[1]

            for idx, cam in enumerate(self.cams):
                table_path = os.path.join(BASE_PATH, f'cam{idx + 1}', 'table.npz')
                np.savez(table_path, table=cam.table)
        else:
            for idx, cam in enumerate(self.cams):
                table_path = os.path.join(BASE_PATH, f'cam{idx + 1}', 'table.npz')
                with np.load(table_path, allow_pickle=True) as f_table:
                    cam.table = f_table['table']

    def next_frame(self):
        voxels = []
        colors = []

        voxels_tmp = []
        # TODO: Look into saving voxels for each frame, then handling only changed voxels, how to do colors?
        for cam in self.cams:
            cam.next_frame()
            # For white pixels in cam.fg
            fg = cam.fg.nonzero()
            voxels_tmp.append([])
            for pix_x, pix_y in zip(fg[0], fg[1]):
                voxels_tmp[cam.idx - 1].extend(cam.table[(pix_y, pix_x)])
        
        voxels = set.intersection(*map(set,voxels_tmp))


        pass

        # foregrounds = []
        # for cam in self.cams:
        #     cam.next_frame()
        #     fg = cam.get_foreground()
        #     foregrounds.append(fg)
        #     pass
        #     # cv.imshow(f'fg cam{cam.idx}', fg)
        #     # cv.imshow(f'img cam{cam.idx}', cam.frame)
        # # cv.waitKey(0)
        # # TODO: Instead of looping through all voxels, loop through pixels in foregrounds
        # for x in range(RESOLUTION):
        #     print(x)
        #     for y in range(RESOLUTION):
        #         for z in range(RESOLUTION):
        #             ret, voxel, color = self.is_in_all_foregrounds((x, y, z), foregrounds)
        #             if ret:
        #                 voxels.append(voxel)
        #                 colors.append(color)

        return voxels, colors

    def is_in_all_foregrounds(self, coords, foregrounds):
        x, y, z = coords
        color = np.zeros((0, 3))
        for cam, fg in zip(self.cams, foregrounds):
            x_y = cam.table[x, y, z, :]
            coord_x, coord_y = tuple(x_y)
            if coord_x >= fg.shape[1] or coord_y >= fg.shape[0] or coord_x < 0 or coord_y < 0 or fg[coord_y, coord_x] == 0:
                return False, None, None
            color = np.vstack([color, (cam.frame[coord_y, coord_x])])

        # Determine voxel positon, added some shifting to ensure placement in the middle of the plane, and facing up.
        voxel = [x-(int(RESOLUTION)/2), -z+RESOLUTION, y-(int(RESOLUTION)/2)]

        # Color in [0,1.0] interval
        color = np.mean(color, axis=0) / 100

        # BGR to RGB for OpenGL
        color[[0, 2]] = color[[2, 0]]

        return True, voxel, color

# Create 3d to 2d lookup table
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