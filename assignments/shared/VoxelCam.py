from shared.BackgroundSubtractor import BackgroundSubtractor
import os, numpy as np, cv2 as cv

class VoxelCam():

    BASE_PATH = os.path.join('assignments', 'data')
    TABLE_SIZE = (100, 100, 100)

    def __init__(self, idx):
        self.idx = idx
        cam_path = os.path.join(self.BASE_PATH, f'cam{idx}')
        background_path = os.path.join(cam_path, 'background.avi')
        video_path = os.path.join(cam_path, 'video.avi')
        config_path = os.path.join(cam_path, 'config.npz')

        self.subtractor = BackgroundSubtractor(background_path)
        with np.load(config_path) as calibration:
            self.mtx = calibration['mtx']
            self.dist = calibration['dist']
            self.rvec = calibration['rvec']
            self.tvec = calibration['tvec']

        self.video = cv.VideoCapture(video_path)

        self.table = np.empty(self.TABLE_SIZE, dtype=tuple)
    
    # Advance the video of this cam by one frame and return it
    def next_frame(self):
        if self.video.isOpened():
            ret, frame = self.video.read()
            if ret:
                return frame
            else:
                return None
    
    # Since objects like cv.VideoCapture and BackgroundSubtractorKNN are not pickle-able,
    # create subset of info needed for creating the lookup table and return it
    def get_info(self):
        return {
            'mtx': self.mtx,
            'dist': self.dist,
            'rvec': self.rvec,
            'tvec': self.tvec,
            'table': self.table
        }
    
    def calc_table(self):
        #     for x in range(cam.table.shape[0]):
#         for y in range(cam.table.shape[1]):
#             for z in range(cam.table.shape[2]):
#                 cam.table[x, y, z] = cv.projectPoints(np.float32([x, y, z]), cam.rvec, cam.tvec, cam.mtx, cam.dist)