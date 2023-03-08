from shared.VoxelReconstructor import VoxelReconstructor
import multiprocessing as mp
import numpy as np

# Testing file for voxel reconstruction debugging, use executable.py
if __name__ == '__main__':
	# Set multiprocessing start method (for windows machines)
    mp.set_start_method('spawn')
    vr = VoxelReconstructor(create_table=False)
    voxels, colors, _ = vr.next_frame()
    # for cm in vr.color_models:
    #     for person, model in cm.models.items():
    #         if person == 3:
    #             print(f'Cam: {cm.cam.idx}, Person: {person}')
    #             print(model.getMeans())
    #             print(model.getCovs())
    #             ret, pred = model.predict(np.float64([[100, 100, 100], [0, 0 ,0], [200, 200, 200], [150, 150, 150]]))
    #             print(np.sum(pred, 0))
    #             print(pred)
    print('hi!')