import os
import cv2 as cv, numpy as np

class ColorModel():
    
    AMOUNT_OF_PEOPLE = 4

    # Cam must always be provided
    def __init__(self, cam, voxels=None):
        # Cam must be at preferred frame position for modelling color
        self.models = {}
        self.cam = cam
        if voxels is None:
            # Load color model from disk
            for person in range(0, self.AMOUNT_OF_PEOPLE):
                path = os.path.join(self.cam.cam_path, f'colormodel_{person}.model')
                self.models[person] = cv.ml.EM.load(path)
            return
        
        # If voxels are provided (offline), create model given cam + voxels
        self.create_models(voxels)

        # Group all coordinates belonging to cluster (person)
        # Create color model (SD + mean or similar), two dicts again? 

    def match_persons(self, voxels):
        labels = np.ravel(self.cluster(voxels))
        np_voxels = np.float32(voxels)

        # For each label
        for label in range(0, self.AMOUNT_OF_PEOPLE):
            p_voxels = np.int_(np_voxels[labels == label])
            colors = np.empty((0, 3))
            # Sample from voxels to check distance in models
            sample = p_voxels[np.random.choice(p_voxels.shape[0], 100, replace=False)]

            # For each cluster (all labeled voxels)
            for v in sample:
                # Find coordinates belonging to voxel
                pixel = self.cam.table_r[v[0], v[1], v[2], :]
                pixel_color = self.cam.frame[pixel[1], pixel[0]]
                colors = np.vstack((colors, pixel_color))
            for person, model in self.models.items():
                ret, preds = model.predict(colors)
                print(f'Label: {label}, Cam {self.cam.idx}, Person {person}:')
                print(np.sum(preds, 0))

    def create_models(self, voxels):
        # Cluster voxels
        labels = np.ravel(self.cluster(voxels))
        np_voxels = np.float32(voxels)

        # Show image and wait for confirmation (keypress)
        cv.imshow(f'{self.cam.idx} img', self.cam.frame)
        cv.waitKey(0)

        # For each label
        for label in range(0, self.AMOUNT_OF_PEOPLE):
            p_voxels = np.int_(np_voxels[labels == label])

            model = cv.ml.EM_create()
            model.setClustersNumber(3)
            colors = np.empty((0, 3))

            # For each cluster (all labeled voxels)
            for v in p_voxels:
                # Find coordinates belonging to voxel
                pixel = self.cam.table_r[v[0], v[1], v[2], :]
                pixel_color = self.cam.frame[pixel[1], pixel[0]]
                colors = np.vstack((colors, pixel_color))
            
            # Create model per person (cluster)
            model.trainEM(colors)
            self.models[label] = model

            # Save model to disk
            path = os.path.join(self.cam.cam_path, f'colormodel_{label}.model')
            model.save(path)

    def cluster(self, voxels):
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # Convert to numpy array and drop height information
        np_voxels = np.float32(voxels)[:,[0,2]]
        _, labels, _ = cv.kmeans(np_voxels, self.AMOUNT_OF_PEOPLE, None, criteria, 10, cv.KMEANS_PP_CENTERS)
        return labels