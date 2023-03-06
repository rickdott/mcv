from shared.VoxelReconstructor import VoxelReconstructor
import multiprocessing as mp
# Testing file for voxel reconstruction debugging, use executable.py
if __name__ == '__main__':
	# Set multiprocessing start method (for windows machines)
    mp.set_start_method('spawn')
    vr = VoxelReconstructor(create_table=True, cams=[1])
    voxels, colors = vr.specific_frame(100)
    