from shared.BackgroundSubtractor import BackgroundSubtractor
import os, numpy as np, cv2 as cv

# So it runs when running from mcv folder and assignments folder
if os.getcwd().endswith('assignments'):
    BASE_PATH = os.path.join('data')
else:
    BASE_PATH = os.path.join('assignments', 'data')


class VoxelCam():

    def __init__(self, idx):
        self.idx = idx
        cam_path = os.path.join(BASE_PATH, f'cam{idx}')
        background_path = os.path.join(cam_path, 'background.avi')
        video_path = os.path.join(cam_path, 'video.avi')
        config_path = os.path.join(cam_path, 'config.npz')

        self.subtractor = BackgroundSubtractor(background_path)
        self.fg = None

        with np.load(config_path) as calibration:
            self.mtx = calibration['mtx']
            self.dist = calibration['dist']
            self.rvec = calibration['rvec']
            self.tvec = calibration['tvec']

        self.video = cv.VideoCapture(video_path)
    
    # Advance the video of this cam by one frame and return it
    def next_frame(self):
        if self.video.isOpened():
            ret, self.frame = self.video.read()
            self.fg = self.subtractor.get_foreground(self.frame) if self.fg is None else cv.bitwise_xor(self.fg, self.subtractor.get_foreground(self.frame))
            return ret
    
    # Since objects like cv.VideoCapture and BackgroundSubtractorKNN are not pickle-able,
    # create subset of info needed for creating the lookup table and return it
    def get_info(self):
        return {
            'mtx': self.mtx,
            'dist': self.dist,
            'rvec': self.rvec,
            'tvec': self.tvec,
            'idx': self.idx
        }
