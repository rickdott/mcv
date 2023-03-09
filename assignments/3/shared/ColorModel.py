import os
import cv2 as cv, numpy as np
from collections import defaultdict

AMOUNT_OF_PEOPLE = 4

class ColorModel():
    
    # Cam must always be provided
    def __init__(self, cam, voxels=None, mean_colors=None):
        # Cam must be at preferred frame position for modelling color
        # label -> (model, predictions) dictionary
        self.models = {}

        # label -> [color] dictionary for matching while creating color models
        self.mean_colors = {} if mean_colors is None else mean_colors
        self.cam = cam
        if voxels is None:
            # Load color model from disk
            for person in range(0, AMOUNT_OF_PEOPLE):
                path = os.path.join(self.cam.cam_path, f'colormodel_{person}.model')
                model = cv.ml.EM.load(path)
                with np.load(os.path.join(self.cam.cam_path, f'colormodel_{person}_preds.npz')) as f:
                    preds = f['preds']
                self.models[person] = (model, preds)
            return
        
        # If voxels are provided (offline), create model given cam + voxels
        self.create_models(voxels)


    def match_persons(self, voxels, labels):
        voxels = np.float32(voxels)
        labels = np.ravel(labels)
        # label > person
        label_map = defaultdict(lambda: 0)
        
        # For every person/cluster
        for label in range(0, AMOUNT_OF_PEOPLE):
            p_voxels = np.int_(voxels[labels == label])
            colors = np.empty((0, 3))

            dists = defaultdict(lambda: 0)

            # Sample from voxels to determine distance to models
            sample = p_voxels[np.random.choice(p_voxels.shape[0], 100, replace=True)]

            # For each cluster (all labeled voxels)
            for v in sample:
                # Find coordinates belonging to voxel
                pixel = self.cam.table_r[v[0], v[1], v[2], :]
                pixel_color = self.cam.frame[pixel[1], pixel[0]]
                colors = np.vstack((colors, pixel_color))
            
            # Compare colors in cluster to all models in current camera
            for person, model in self.models.items():
                _, new_preds = model[0].predict(colors)
                mean_pred = np.mean(new_preds, axis=0)
                # Compare mean prediction to mean prediction of color model, defined as the L2/Euclidean distance
                dist = np.linalg.norm(mean_pred - model[1])
                dists[person] += dist
            # Here dists is what one cam thinks of all voxels
            # Weight differently based on std?
            person, dist = min(dists.items(), key=lambda x: x[1])
            label_map[label] = (person, dist)

        return dict(label_map)
        # Should return label > person map
        # Include person > color map

    def create_models(self, voxels):
        # Get voxels + labels that should be used for creating a color model (those above mean height)
        voxels, labels = remove_pants(voxels, cluster(voxels))
        labels = np.ravel(labels)
        voxels = np.float32(voxels)

        # For every person/cluster known
        for label in range(0, AMOUNT_OF_PEOPLE):
            p_voxels = np.int_(voxels[labels == label])
            colors = np.empty((0, 3))

            # Create a model with 3 Gaussian clusters
            model = cv.ml.EM_create()
            model.setClustersNumber(3)
            
            # For each cluster (group of labelled voxels
            for v in p_voxels:
                # Find coordinates belonging to voxel
                pixel = self.cam.table_r[v[0], v[1], v[2], :]
                pixel_color = self.cam.frame[pixel[1], pixel[0]]
                colors = np.vstack((colors, pixel_color))
            # Create model per person (cluster)
            model.trainEM(colors)

            # Get probability distribution of person under model, in online phase the distribution closest to this one corresponds to the person
            _, preds = model.predict(colors)
            preds = np.mean(preds, axis=0)

            # Use mean color to match model + label combinations to other cameras
            mean_color = np.mean(colors, axis=0)
            # Determine new label, this label should be the label that has the least L2/Euclidean distance to previous mean colors
            if len(self.mean_colors) == AMOUNT_OF_PEOPLE:
                new_label, _ = min(self.mean_colors.items(), key=lambda old_mean_color: np.linalg.norm(old_mean_color[1] - mean_color))
            else:
                new_label = label

            self.models[new_label] = (model, preds)
            self.mean_colors[new_label] = mean_color

            # Save model and preds to disk
            model.save(os.path.join(self.cam.cam_path, f'colormodel_{new_label}.model'))
            np.savez(os.path.join(self.cam.cam_path, f'colormodel_{new_label}_preds.npz'), preds=preds)

def cluster(voxels):
    # Cluster voxels in 3d space based on x/y information
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.1)

    # Convert to numpy array and drop height information
    np_voxels = np.float32(voxels)[:,[0,1]]

    _, labels, _ = cv.kmeans(np_voxels, AMOUNT_OF_PEOPLE, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS)
    return labels

def remove_pants(voxels, labels):
    # From clustered voxels + labels, select only those voxels
    # and labels that are below the belt
    labels, voxels = np.array(labels), np.array(voxels)
    mean_height = np.mean(voxels[:, 2], dtype=np.int_)

    # Subset voxels based on mean height (reversed, so < means above)
    expr = voxels[:, 2] < mean_height
    return voxels[expr], labels[expr]