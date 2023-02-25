import numpy as np, cv2 as cv
from shared.VoxelCam import VoxelCam, TABLE_SIZE
import multiprocessing as mp

class VoxelReconstructor():

    def __init__(self):
        # Create VoxelCam instances and pre-load their pickle-able information sets
        self.cams = []
        self.cam_infos = []
        for cam in range(1, 5):
            vcam = VoxelCam(cam)
            self.cams.append(vcam)
            self.cam_infos.append(vcam.get_info())

        # Parallelized calculation of lookup table
        with mp.Pool(len(self.cams)) as p:
            results = p.map(calc_table, self.cam_infos)
        
        for result in results:
            self.cams[result[0] - 1].table = result[1]

    def next_frame(self):
        voxels = []

        foregrounds = []
        # TODO: Not working,returns empty list :(
        for cam in self.cams:
            cam.next_frame()
            foregrounds.append(cam.get_foreground())
        for x in range(TABLE_SIZE[0]):
            for y in range(TABLE_SIZE[1]):
                for z in range(TABLE_SIZE[2]):
                    if self.is_in_all_foregrounds((x, y, z), foregrounds):
                        voxels.append([x, y, z])
        return voxels

    def is_in_all_foregrounds(self, coords, foregrounds):
        x, y, z = coords
        for cam, fg in zip(self.cams, foregrounds):
            coord_x, coord_y = cam.table[x, y, z]
            # TODO: Maybe store as int originally?
            if fg[int(coord_x), int(coord_y)] == 0:
                return False
        return True

# Create 3d to 2d lookup table
# use cv.projectPoints to project all points in needed (X, Y, Z) space to (X, Y) space for each camera
def calc_table(cam):
    print(f'Calculating table for {cam["idx"]}')
    table = np.empty(TABLE_SIZE, dtype=tuple)
    for x in range(table.shape[0]):
        for y in range(table.shape[1]):
            for z in range(table.shape[2]):
                table[x, y, z] = tuple(cv.projectPoints(np.float32([x, y, z]), cam['rvec'], cam['tvec'], cam['mtx'], cam['dist'])[0][0][0])
    return (cam['idx'], table)