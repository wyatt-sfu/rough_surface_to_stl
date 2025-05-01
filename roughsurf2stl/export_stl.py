import struct
import numpy as np
import tqdm

HEADER_SIZE = 80  # bytes
NUM_TRI_FORMAT = "<I"
TRIANGLE_FORMAT = "<12fh"


def export_stl(triangles: np.ndarray, normals: np.ndarray, out_name: str):
    """
    Creates a binary STL file with the given triangles. STL files contain no
    information on units, so the coordinates are assumed to already be in the
    desired units. Using the STL file definition found on Wikipedia here:
    https://en.wikipedia.org/wiki/STL_(file_format).

    Args:
        triangles (np.ndarray): An array of shape (n_triangles, 3, 3). Each triangle
            is defined by 3 vertices, with each vertex having (x, y, z) coordinates.
              Triangle definition : | V1_x, V1_y, V1_z |
                                    | V2_x, V2_y, V2_z |
                                    | V3_x, V3_y, V3_z |
        normals (np.ndarray): An array of shape (n_triangles, 3) containing a vector
            normal to each triangle. Last dimension is ordered (x, y, z).
        out_name (str): Filename of the output STL file (should end with .stl)
    """
    n_tri = triangles.shape[0]
    tri_struct = struct.Struct(TRIANGLE_FORMAT)

    header = bytearray(HEADER_SIZE)

    with open(out_name, mode="wb") as stl_file:
        # Write out the header info to the STL file
        stl_file.write(header)
        stl_file.write(struct.pack(NUM_TRI_FORMAT, n_tri))

        for i in tqdm.trange(n_tri):
            # Write the triangle to the file
            stl_file.write(
                tri_struct.pack(
                    TRIANGLE_FORMAT,
                    normals[i, 0],
                    normals[i, 1],
                    normals[i, 2],
                    triangles[i, 0, 0],
                    triangles[i, 0, 1],
                    triangles[i, 0, 2],
                    triangles[i, 1, 0],
                    triangles[i, 1, 1],
                    triangles[i, 1, 2],
                    triangles[i, 2, 0],
                    triangles[i, 2, 1],
                    triangles[i, 2, 2],
                    0,
                )
            )
