import numpy as np, cv2 as cv
from shared.VoxelCam import VoxelCam, TABLE_SIZE, BASE_PATH
import multiprocessing as mp
import os

class VoxelReconstructor():

    RESOLUTION = 100

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

        foregrounds = []
        for cam in self.cams:
            cam.next_frame()
            fg = cam.get_foreground()
            foregrounds.append(fg)
        #     cv.imshow(f'fg cam{cam.idx}', fg)
        # cv.waitKey(0)
        for x in range(self.RESOLUTION):
            print(x)
            for y in range(self.RESOLUTION):
                for z in range(self.RESOLUTION):
                    if self.is_in_all_foregrounds((x, y, z), foregrounds):
                        # voxels.append([x, z, y])
                        voxels.append([x-(int(self.RESOLUTION)/2), -z+self.RESOLUTION, y-(int(self.RESOLUTION)/2)])
                        colors.append([x/100, z/100, y/100])
                        # voxels.append([x-int(self.RESOLUTION / 2), z, y-int(self.RESOLUTION / 2)])
                        # Y is the UP DIRECTION
        return voxels, colors

    def is_in_all_foregrounds(self, coords, foregrounds):
        x, y, z = coords
        for cam, fg in zip(self.cams, foregrounds):
            coord_x, coord_y = tuple(cam.table[x, y, z, :])
            if coord_x >= fg.shape[1] or coord_y >= fg.shape[0]:
                return False
            if coord_x < 0 or coord_y < 0:
                return False
            if fg[coord_y, coord_x] == 0:
                return False
            # else:
            #     print('Pixel found in fg') 
            # print(coord_x, coord_y)
            # print(fg[int(coord_x), int(coord_y)])

            # if fg[coord_y, coord_x] == 255:
            #     # print(cam.idx, coord_x, coord_y)
            #     if cam.idx == 4:
            #         return True
        return True

# Create 3d to 2d lookup table
# use cv.projectPoints to project all points in needed (X, Y, Z) space to (X, Y) space for each camera
def calc_table(cam):
    # TODO: Cant be correct, table[0, 0, 0] should be coordinates of origin in world
    # Check using projectpoints which point should correspond to [0,0,0], then change reshaping until that ocordinate is there
    print(f'{cam["idx"]} Calculating table')
    # Create grid of all possible index combinations
    # One block/one step is 115mm, local resolution since this function is parallelized
    steps = 100
    steps_c = complex(steps)
    # grid = np.float32(np.mgrid[-TABLE_SIZE[0]:TABLE_SIZE[0]:steps, -TABLE_SIZE[1]:TABLE_SIZE[1]:steps, -TABLE_SIZE[2]*2:0:steps])
    grid = np.float32(np.mgrid[-400:1100:steps_c, -750:750:steps_c, -1500:0:steps_c])
    # grid = np.float32(np.mgrid[-TABLE_SIZE[0]:TABLE_SIZE[0], -TABLE_SIZE[1]:TABLE_SIZE[1], 0:TABLE_SIZE[2]])
    # grid = np.float32(np.meshgrid(np.arange(0, TABLE_SIZE[0]), np.arange(0, TABLE_SIZE[1]), np.arange(0, TABLE_SIZE[2])))
    grid_t = grid.T.reshape(-1, 3)
    # grid_t = grid.T.reshape(-1, 3) * 115

    print(f'{cam["idx"]} Projecting points')
    # Project indices to 2d
    proj_list = cv.projectPoints(grid_t, cam['rvec'], cam['tvec'], cam['mtx'], cam['dist'])[0]

    # Create table of (X, Y, Z, 2) shape containing 2d coordinates
    # table = np.int_(proj_list).reshape(list(TABLE_SIZE) + [2], order='F')
    table = np.int_(proj_list).reshape((steps, steps, steps, 2), order='F')

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