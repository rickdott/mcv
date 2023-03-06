import os
import cv2 as cv, numpy as np

class ColorModel():
    
    AMOUNT_OF_PEOPLE = 4

    def __init__(self, cam=None, voxels=None, online=False):
        # Cam must be at preferred frame position for modelling color
        cm_path = os.path.join(cam.cam_path, 'colormodel.npz')

        if online:
            # Load color model from disk
            return
        
        self.cam = cam
        self.voxels = voxels

        # Cluster voxels
        labels = np.ravel(self.cluster(voxels))
        np_voxels = np.float32(voxels)

        # For each label
        for label in range(0, self.AMOUNT_OF_PEOPLE):
            p_voxels = np.int_(np_voxels[labels == label])
            for v in p_voxels:
                pixel = cam.table_r[v[0], v[2], v[1], :]
                pixel_color = cam.frame[pixel[1], pixel[0]]



        # For each cluster (all labeled voxels)

        # Find coordinates belonging to voxel (loop through cam.table until voxel in voxels)

        # Group all coordinates belonging to cluster (person)
        # Create color model (SD + mean or similar), two dicts again? 

    def create_model(self):
        pass

    def cluster(self, voxels):
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # Convert to numpy array and drop height information
        np_voxels = np.float32(voxels)[:,[0,2]]
        _, labels, _ = cv.kmeans(np_voxels, self.AMOUNT_OF_PEOPLE, None, criteria, 10, cv.KMEANS_PP_CENTERS)
        return labels