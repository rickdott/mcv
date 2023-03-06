from shared.ColorModel import ColorModel
from shared.VoxelReconstructor import VoxelReconstructor
import multiprocessing as mp
# Testing file for voxel reconstruction debugging, use executable.py
if __name__ == '__main__':
	# Set multiprocessing start method (for windows machines)
    mp.set_start_method('spawn')
    vr = VoxelReconstructor(create_table=True)
    voxels, _ = vr.specific_frame(100)
    cm = ColorModel(vr.cams[0], voxels)