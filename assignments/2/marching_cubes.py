"""
==============
Marching Cubes
==============

Marching cubes is an algorithm to extract a 2D surface mesh from a 3D volume.
This can be conceptualized as a 3D generalization of isolines on topographical
or weather maps. It works by iterating across the volume, looking for regions
which cross the level of interest. If such regions are found, triangulations
are generated and added to an output mesh. The final result is a set of
vertices and a set of triangular faces.

The algorithm requires a data volume and an isosurface value. For example, in
CT imaging Hounsfield units of +700 to +3000 represent bone. So, one potential
input would be a reconstructed CT set of data and the value +700, to extract
a mesh for regions of bone or bone-like density.

This implementation also works correctly on anisotropic datasets, where the
voxel spacing is not equal for every spatial dimension, through use of the
`spacing` kwarg.

Adapted from https://scikit-image.org/docs/stable/auto_examples/edges/plot_marching_cubes.html
For use in an assignment in the Computer Vision (2023) course at Universiteit Utrecht
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure

from shared.VoxelReconstructor import VoxelReconstructor, RESOLUTION

import multiprocessing as mp
if __name__ == '__main__':
	# Set multiprocessing start method (for windows machines)
    mp.set_start_method('spawn')
    vr = VoxelReconstructor(create_table=True)

    # Get voxels belonging to the first frame
    fr = vr.next_frame()[0]
    arr = np.zeros((RESOLUTION, RESOLUTION, RESOLUTION))

    # Offsets assuming a resolution of 50, recommend changing VoxelReconstructor.RESOLUTION to 50 before running
    # higher values are not performant
    for pos in fr:
        arr[int(pos[0] + 20), int(pos[2] + 15),  int(pos[1])] = 1

    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, normals, values = measure.marching_cubes(arr, 0)

    # Display resulting triangular mesh using Matplotlib
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")
    ax.set_aspect('equal')
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_zlim(0, 45)

    plt.tight_layout()
    plt.show()
