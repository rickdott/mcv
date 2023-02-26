import numpy as np, cv2 as cv
from shared.VoxelCam import VoxelCam, TABLE_SIZE, BASE_PATH
import multiprocessing as mp
import os

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
        cube_points = np.uint8([[0,0,0], [0,TABLE_SIZE[1]-1,0], [TABLE_SIZE[0]-1,TABLE_SIZE[1]-1,0], [TABLE_SIZE[0]-1,0,0],
                   [0,0,TABLE_SIZE[2]-1], [0,TABLE_SIZE[1]-1,TABLE_SIZE[2]-1], [TABLE_SIZE[0]-1,TABLE_SIZE[1]-1,TABLE_SIZE[2]-1], [TABLE_SIZE[0]-1,0,TABLE_SIZE[2]-1]])
        foregrounds = []
        for cam in self.cams:
            cam.next_frame()
            foregrounds.append(cam.get_foreground())
            for point in cube_points:

                point2d = cam.table[point[0], point[1], point[2]].astype(np.uint8)
                cv.circle(cam.frame, point2d, 4, (0, 0, 255), thickness=cv.FILLED)
            cv.imshow(f'cam{cam.idx}', cam.frame)
        cv.waitKey(0)
        # Visualize fg
        # for i, fg in enumerate(foregrounds):
        #     # cv.imshow(str(i), fg)
        #     cv.drawFrameAxes(self.cams[i].frame, self.cams[i].mtx, self.cams[i].dist, self.cams[i].rvec, self.cams[i].tvec, 1000, 1)
        #     cv.imshow(f'img: {i}', self.cams[i].frame)
        # cv.waitKey(0)
        for x in range(TABLE_SIZE[0]):
            for y in range(TABLE_SIZE[1]):
                for z in range(TABLE_SIZE[2]):
                    if self.is_in_all_foregrounds((x, y, z), foregrounds):
                        voxels.append([x, y, z])
        return voxels

    def is_in_all_foregrounds(self, coords, foregrounds):
        x, y, z = coords
        for cam, fg in zip(self.cams, foregrounds):
            coord_x, coord_y = tuple(cam.table[x, y, z, :])
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
    print(f'Calculating table for {cam["idx"]}')
    # Create grid of all possible index combinations
    # Multiply with cell_size necessary? Not doing so results in even smaller extent
    grid = np.float32(np.meshgrid(np.arange(0, TABLE_SIZE[0]), np.arange(0, TABLE_SIZE[1]), np.arange(0, TABLE_SIZE[2])))
    grid = grid.T.reshape(-1, 3)

    # Project indices to 2d
    proj_list = cv.projectPoints(grid, cam['rvec'], cam['tvec'], cam['mtx'], cam['dist'])[0]

    # Create table of (X, Y, Z, 2) shape containing 2d coordinates
    table = np.uint8(proj_list).reshape(list(TABLE_SIZE) + [2], order='F')
    # table_t = np.transpose(table, (1, 0, 2, 3))
    return (cam['idx'], table)