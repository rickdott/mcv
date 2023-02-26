from shared.VoxelReconstructor import VoxelReconstructor
import multiprocessing as mp
if __name__ == '__main__':
	# set the start method
    mp.set_start_method('spawn')
    vr = VoxelReconstructor(create_table=True)
    fr = vr.next_frame()
    print('hi!')