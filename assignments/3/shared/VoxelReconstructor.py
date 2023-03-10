import numpy as np, cv2 as cv
from shared.ColorModel import ColorModel, cluster, remove_pants
from shared.VoxelCam import VoxelCam, BASE_PATH
import multiprocessing as mp
import os
from collections import defaultdict, Counter

# Resolution to use when generating the voxel model
RESOLUTION = 50

EXTENT_XMIN = -700
EXTENT_XMAX = 2500
EXTENT_YMIN = -3500
EXTENT_YMAX = 1500
EXTENT_ZMIN = -2000
EXTENT_ZMAX = 0
# The VoxelReconstructor class handles calculating the lookup tables for individual VoxelCams
# and gathering the voxel, color combinations for the next frame.
class VoxelReconstructor():

    def __init__(self, create_table=False, cams=[1, 2, 3, 4], use_color_model=True):
        # Create VoxelCam instances and pre-load their pickle-able information sets
        self.cams = []
        self.cam_infos = []
        self.use_color_model = use_color_model
        self.color_models = []
        self.person_to_color = {0: np.array([1, 0, 0]), 1: np.array([0, 1, 0]), 2: np.array([0, 0, 1]), 3: np.array([1, 0, 1])}

        for cam in cams:
            vcam = VoxelCam(cam)
            self.cams.append(vcam)
            self.cam_infos.append(vcam.get_info())
            if self.use_color_model:
                self.color_models.append(ColorModel(vcam))
        
        self.cam_amount = len(self.cams)

        # (X, Y, Z) -> count, color dicts
        self.voxels = defaultdict(lambda: [False] * self.cam_amount)
        self.colors = defaultdict(lambda: [[0, 0, 0]] * self.cam_amount)

        if create_table:
            # Parallelized calculation of lookup table
            with mp.Pool(len(self.cams)) as p:
                results = p.map(calc_table, self.cam_infos)
            
            for result in results:
                # Table of coordinates to voxels
                self.cams[result[0] - 1].table = result[1]
                # Table of voxel to coordinates
                self.cams[result[0] - 1].table_r = result[2]

            # Save tables to disk
            for idx, cam in enumerate(self.cams):
                table_path = os.path.join(BASE_PATH, f'cam{idx + 1}', 'table.npz')
                np.savez(table_path, table=cam.table, table_r=cam.table_r)
        else:
            for idx, cam in enumerate(self.cams):
                table_path = os.path.join(BASE_PATH, f'cam{idx + 1}', 'table.npz')
                with np.load(table_path, allow_pickle=True) as f_table:
                    cam.table = f_table['table'].item()
                    cam.table_r = f_table['table_r']

    # Selects the changed voxels + colors for the next frame
    def next_frame(self):
        next_voxels = []
        orig_voxels = []
        next_colors = []

        # For every cam/view, advance it one frame and receive the changed pixels
        for cam in self.cams:
            ret = cam.next_frame()
            if not ret:
                return
            changed = cam.xor.nonzero()

            print(f'{cam.idx} Changed pixels: {len(changed[0])}')

            # For every changed foreground pixel, increment its connected voxels' voxel counters by one
            # also add the color of the foreground pixel for the current camera
            for pix_y, pix_x in zip(changed[0], changed[1]):
                coord = (pix_x, pix_y)
                for voxel in cam.table[coord]:
                    # If voxel was shown in previous frame, should be visible in all 4 views this time
                    self.voxels[voxel][cam.idx-1] = cam.fg[pix_y, pix_x] != 0
                    if cam.fg[pix_y, pix_x] != 0:
                        self.colors[voxel][cam.idx-1] = cam.frame[pix_y, pix_x]

        # For all the voxel, color combinations, add those that occurred in all cameras
        for voxel, check in self.voxels.items():
            if all(check):
                # Add voxel including offsetting due to calibration and to place it in the middle of the plane
                next_voxels.append([voxel[0]-(int(RESOLUTION)/2), -voxel[2]+RESOLUTION, voxel[1]-(int(RESOLUTION)/2)])
                orig_voxels.append(voxel)

                # Divide by 100 to get color in [0, 1] interval
                color = np.mean(np.array(self.colors[voxel]), axis=0) / 100

                # BGR to RGB (for OpenGL)
                color[[0, 2]] = color[[2, 0]]
                
                next_colors.append(color)

        # If color model is used, cluster voxels of interest (above the waist) and compare to predictions of similarly defined color models
        if self.use_color_model:
            # Labels of all voxels
            labels, centers = cluster(orig_voxels)

            ratio_x = (abs(EXTENT_XMIN) + abs(EXTENT_XMAX)) / RESOLUTION
            ratio_y = (abs(EXTENT_YMIN) + abs(EXTENT_YMAX)) / RESOLUTION

            centers[:, 0] = EXTENT_XMIN + (centers[:, 0] * ratio_x)
            centers[:, 1] = EXTENT_YMIN + (centers[:, 1] * ratio_y)

            # Voxels and labels needed for comparison
            cluster_voxels, cluster_labels = remove_pants(orig_voxels, labels)

            # cam > label > person
            lp_mat = np.ones((4,4,4))

            for cam in self.cams:
                self.color_models[cam.idx - 1].match_persons(cluster_voxels, centers, cluster_labels, lp_mat)
            print(lp_mat)

            average = np.nanmean(lp_mat, axis=0)
            print(average)
            print(average.argmin(axis=1))

            label_person = dict(zip(range(0,4), average.argmin(axis=1)))

            # Create dict person > label with lowest values
            # maxes = list(np.argmin(lp_mat, axis=0))
            # print(maxes)

            next_colors = [list((np.array(color) + self.person_to_color[label_person[label.item()]]) / 2) for color, label in zip(next_colors, labels)]
            # next_colors = [list(self.person_to_color[person_label[label.item()]]) for label in labels]
                    
        # colors = ((labels + 1) / 4).tolist()
        # next_colors = [[color, 0, 0] for color in colors]
        return next_voxels, next_colors, orig_voxels

    def specific_frame(self, frame_index):
        # For list of indices of length equal to amount of cameras, get frames for those videos
        for cam in self.cams:
            cam.video.set(cv.CAP_PROP_POS_FRAMES, frame_index)
        return self.next_frame()

# Create (X, Y) -> [(X, Y, Z), ...] dictionary
# use cv.projectPoints to project all points in needed (X, Y, Z) space to (X, Y) space for each camera
def calc_table(cam):
    print(f'{cam["idx"]} Calculating table')
    steps_c = complex(RESOLUTION)
    # Create evenly spaced grid centered around the subject, using n = RESOLUTION steps.
    grid = np.float32(np.mgrid[-700:2500:steps_c, -3500:1500:steps_c, -2000:0:steps_c])
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
    return (cam['idx'], table_d, table)