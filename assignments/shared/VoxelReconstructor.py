import numpy as np, cv2 as cv
from shared.VoxelCam import VoxelCam, TABLE_SIZE, BASE_PATH
import multiprocessing as mp
import os
import math

class VoxelReconstructor():

    RESOLUTION = 128

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
        # TABLE_SIZE = np.array((128, 128, 64)) * 115
        mm = 2000
        # cube_points = np.int_([[0,0,0], [0,TABLE_SIZE[1]-1,0], [TABLE_SIZE[0]-1,TABLE_SIZE[1]-1,0], [TABLE_SIZE[0]-1,0,0],
                #    [0,0,TABLE_SIZE[2]-1], [0,TABLE_SIZE[1]-1,TABLE_SIZE[2]-1], [TABLE_SIZE[0]-1,TABLE_SIZE[1]-1,TABLE_SIZE[2]-1], [TABLE_SIZE[0]-1,0,TABLE_SIZE[2]-1]])
        cube_points = np.float32([[0,0,0], [0,mm,0], [mm,mm,0], [mm,0,0],
                   [0,0,-mm], [0,mm,-mm], [mm,mm,-mm], [mm,0,-mm]])
        foregrounds = []
        for cam in self.cams:
            cam.next_frame()
            foregrounds.append(cam.get_foreground())
            # projected_points = cv.projectPoints(np.int_([[0,0,0], [0, 2000, 0], [2000, 2000, 0], [2000, 2000, 2000], [2000, 0, 0], [0, 0, -2000]]).astype(np.float32), cam.rvec, cam.tvec, cam.mtx, cam.dist)
            projected_points = cv.projectPoints(cube_points, cam.rvec, cam.tvec, cam.mtx, cam.dist)
            for point in projected_points[0]:
                cv.circle(cam.frame, (int(point[0][0]), int(point[0][1])), 4, (0, 255, 0), thickness=cv.FILLED)
        #     cv.drawFrameAxes(cam.frame, cam.mtx, cam.dist, cam.rvec, cam.tvec, 115*10, 1)
        #     # Instead of using cam.table, try to project points and see difference
        #     for point in cube_points:
        #         # Points should really line up with points in frame axes after projection
        #         point2d = cam.table[point[0], point[1], point[2]].astype(np.int_)
        #         cv.circle(cam.frame, point2d, 4, (0, 0, 255), thickness=cv.FILLED)

            cv.imshow(f'cam{cam.idx}', cam.frame)
        cv.waitKey(0)
        # Visualize fg
        # for i, fg in enumerate(foregrounds):
        #     # cv.imshow(str(i), fg)
            # cv.drawFrameAxes(self.cams[i].frame, self.cams[i].mtx, self.cams[i].dist, self.cams[i].rvec, self.cams[i].tvec, 1000, 1)
            # cv.imshow(f'img: {i}', self.cams[i].frame)
        # cv.waitKey(0)
        # for x in range(-TABLE_SIZE[0], TABLE_SIZE[0]):
        #     print(x)
        #     for y in range(-TABLE_SIZE[1], TABLE_SIZE[1]):
        #         for z in range(TABLE_SIZE[2]):
        #             if self.is_in_all_foregrounds((x, y, z), foregrounds):
        #                 voxels.append([x, y, z])
        for x in range(self.RESOLUTION):
            print(x)
            for y in range(self.RESOLUTION):
                for z in range(self.RESOLUTION):
                    if self.is_in_all_foregrounds((x, y, z), foregrounds):
                        voxels.append([x-64, y, z-64])
                        # Y is the UP DIRECTION
        return voxels

    def is_in_all_foregrounds(self, coords, foregrounds):
        x, y, z = coords
        for cam, fg in zip(self.cams, foregrounds):
            coord_x, coord_y = tuple(cam.table[x, y, z, :])
            if coord_x >= fg.shape[0] or coord_y >= fg.shape[1]:
                return False
            if coord_x < 0 or coord_y < 0:
                return False
            # if fg[int(coord_x), int(coord_y)] == 0:
            #     return False
            # else:
            #     print('Pixel found in fg') 
            # print(coord_x, coord_y)
            # print(fg[int(coord_x), int(coord_y)])

            if fg[coord_x, coord_y] == 255:
                # print(cam.idx, coord_x, coord_y)

                return True
        return False

# Create 3d to 2d lookup table
# use cv.projectPoints to project all points in needed (X, Y, Z) space to (X, Y) space for each camera
def calc_table(cam):
    # TODO: Cant be correct, table[0, 0, 0] should be coordinates of origin in world
    # Check using projectpoints which point should correspond to [0,0,0], then change reshaping until that ocordinate is there
    print(f'{cam["idx"]} Calculating table')
    # Create grid of all possible index combinations
    # Multiply with cell_size necessary? Not doing so results in even smaller extent
    # grid = np.float32(np.mgrid[0:TABLE_SIZE[0], 0:TABLE_SIZE[1], 0:TABLE_SIZE[2]])

    # One block/one step is 115mm
    steps = 128j
    # grid = np.float32(np.mgrid[-TABLE_SIZE[0]:TABLE_SIZE[0]:steps, -TABLE_SIZE[1]:TABLE_SIZE[1]:steps, -TABLE_SIZE[2]*2:0:steps])
    grid = np.float32(np.mgrid[-1000:1000:steps, -1000:1000:steps, -2000:0:steps])
    # grid = np.float32(np.mgrid[-TABLE_SIZE[0]:TABLE_SIZE[0], -TABLE_SIZE[1]:TABLE_SIZE[1], 0:TABLE_SIZE[2]])
    # grid = np.float32(np.meshgrid(np.arange(0, TABLE_SIZE[0]), np.arange(0, TABLE_SIZE[1]), np.arange(0, TABLE_SIZE[2])))
    grid_t = grid.T.reshape(-1, 3) * 115

    print(f'{cam["idx"]} Projecting points')
    # Project indices to 2d
    proj_list = cv.projectPoints(grid_t, cam['rvec'], cam['tvec'], cam['mtx'], cam['dist'])[0]

    # Create table of (X, Y, Z, 2) shape containing 2d coordinates
    # table = np.int_(proj_list).reshape(list(TABLE_SIZE) + [2], order='F')
    table = np.int_(proj_list).reshape((128, 128, 128, 2), order='F')

    # if cam['idx'] == 1:
    #     proj_0 = cv.projectPoints(np.float32([[0,0,0]]), cam['rvec'], cam['tvec'], cam['mtx'], cam['dist'])[0]
    #     proj_10 = cv.projectPoints(np.float32([[10,10,10]]) , cam['rvec'], cam['tvec'], cam['mtx'], cam['dist'])[0]
    #     proj_12 = cv.projectPoints(np.float32([[12,14,16]]), cam['rvec'], cam['tvec'], cam['mtx'], cam['dist'])[0]
    #     proj_140 = cv.projectPoints(np.float32([[111,120,54]]), cam['rvec'], cam['tvec'], cam['mtx'], cam['dist'])[0]
    #     print(table[0, 0, 0,:])
    #     print(table[10, 10, 10,:])
    #     print(table[12, 14, 16,:])
    #     print(table[111, 120, 54,:])
    #     pass
    print(f'{cam["idx"]} Wrapping up')
    return (cam['idx'], table)