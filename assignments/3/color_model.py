from shared.ColorModel import ColorModel
from shared.VoxelReconstructor import VoxelReconstructor
import multiprocessing as mp
# Testing file for voxel reconstruction debugging, use executable.py
if __name__ == '__main__':
	# Set multiprocessing start method (for windows machines)
    mp.set_start_method('spawn')
    vr = VoxelReconstructor(create_table=False)
    _, _, voxels = vr.specific_frame(55)
    cm = ColorModel(vr.cams[0], voxels)

    _, _, voxels = vr.specific_frame(1)
    cm = ColorModel(vr.cams[1], voxels)

    _, _, voxels = vr.specific_frame(540)
    cm = ColorModel(vr.cams[2], voxels)

    _, _, voxels = vr.specific_frame(900)
    cm = ColorModel(vr.cams[3], voxels)