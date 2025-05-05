import numpy as np
import scipy.spatial
import tqdm


def triangulate_surface(
    nx: int, ny: int, dx: float, surf_height: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Takes an array of points defining the surface height and exports it as
    an STL file.

    Args:
        nx (int): Number of sample points in X direction
        ny (int): Number of sample points in Y direction
        dx (float): Distance between samples (assumed square pixels)
        surf_height (np.ndarray): Numpy array of surface height with shape (ny, nx)

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of numpy arrays containing
            (triangles, normals). The shape of the triangles array is
            (ntriangles, 3, 3). Each triangle is defined by a 3x3 matrix in
            which each row is a vector containing the vertex position, ie
                            | V1_x, V1_y, V1_z |
                            | V2_x, V2_y, V2_z |
                            | V3_x, V3_y, V3_z |
            The normals vector has shape (ntriangles, 3) and contains an outward
            facing unit normal vector for each triangle. The last dimension is
            ordered [n_x, n_y, n_z].
    """
    # Compute the XY coordinates of the surface
    x_points = np.arange(nx) * dx
    y_points = np.arange(ny) * dx

    # Create a numpy array holding (x, y) indices of each point on the surface
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    surf_xy_indices = np.stack([xx, yy], axis=-1)

    # Reshape into (nx * ny, 2). Last dimension is the (x, y) index of the point.
    surf_xy_indices = np.reshape(surf_xy_indices, [nx * ny, 2])

    # Triangulate the surface using Delaunay tesselation
    print("Triangulating the XY plane...")
    xy_triangles = scipy.spatial.Delaunay(surf_xy_indices)
    print("... Complete")
    vertex_indices = xy_triangles.simplices
    triangles = _triangle_coord_lookup(
        vertex_indices, surf_xy_indices, x_points, y_points, surf_height
    )
    surf_norm = _triangle_normals(triangles)
    return triangles, surf_norm


def _triangle_coord_lookup(
    vertex_indices: np.ndarray,
    surf_xy_indices: np.ndarray,
    x_points: np.ndarray,
    y_points: np.ndarray,
    surf_height: np.ndarray,
) -> np.ndarray:
    """Create an array of triangle vertex coordinates from indices"""
    n_triangles = vertex_indices.shape[0]
    triangles = np.zeros((n_triangles, 3, 3))

    for i in tqdm.trange(n_triangles, desc="Triangle vertices:"):
        _tri = np.zeros((3, 3))  # Each row is the x,y,z coordinates of a vertex
        for j in range(3):
            # Loop over each vertex in the triangle
            _x_idx = surf_xy_indices[vertex_indices[i, j]][0]
            _y_idx = surf_xy_indices[vertex_indices[i, j]][1]
            _tri[j, 0] = x_points[_x_idx]
            _tri[j, 1] = y_points[_y_idx]
            _tri[j, 2] = surf_height[_y_idx, _x_idx]

        triangles[i, ...] = _tri

    return triangles


def _triangle_normals(triangles: np.ndarray) -> np.ndarray:
    """Compute a normal unit vector to each triangle."""
    n_triangles = triangles.shape[0]
    normals = np.zeros((n_triangles, 3))

    for i in tqdm.trange(n_triangles, desc="Triangle normals:"):
        v1 = triangles[i, 0, :]
        v2 = triangles[i, 1, :]
        v3 = triangles[i, 2, :]
        n_hat = np.cross((v2 - v1), (v3 - v1))
        if n_hat[2] < 0.0:
            # The vertices were not arranged counter-clockwise.
            n_hat = np.cross((v3 - v1), (v2 - v1))
        n_hat = n_hat / np.linalg.norm(n_hat)
        normals[i, :] = n_hat

    return normals
