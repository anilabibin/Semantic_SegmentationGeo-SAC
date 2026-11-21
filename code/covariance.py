
import numpy as np
from sklearn.neighbors import KDTree


def build_kdtree(points):
    """
    Build KDTree for efficient neighbourhood searching.
    """

    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must have shape (N, 3).")

    if len(points) < 3:
        raise ValueError("At least 3 points are required.")

    if not np.isfinite(points).all():
        raise ValueError("Point cloud contains NaN or infinite values.")

    return KDTree(points)


def get_neighbors(points, tree, k=20):
    """
    Find k nearest neighbours for every point.
    """

    k = min(k, len(points))

    distances, indices = tree.query(points, k=k)

    return distances, indices


def calculate_covariance(points, neighbor_indices):
    """
    Calculate local covariance matrices and eigenvalues.

    Returns:
        covariance_matrices: (N, 3, 3)
        eigenvalues: (N, 3), descending order
    """

    neighbors = points[neighbor_indices]

    # Local mean
    mean_points = np.mean(neighbors, axis=1, keepdims=True)

    # Center neighbourhood coordinates
    centered = neighbors - mean_points

    # Calculate covariance matrices
    covariance_matrices = np.einsum(
        'nki,nkj->nij',
        centered,
        centered
    ) / neighbors.shape[1]

    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvalsh(covariance_matrices)

    # Sort in descending order
    eigenvalues = eigenvalues[:, ::-1]

    # Avoid small negative numerical values
    eigenvalues = np.maximum(eigenvalues, 0.0)

    return covariance_matrices, eigenvalues


def compute_local_covariance(points, k=20):
    """
    Complete local covariance computation.
    """

    points = np.asarray(points, dtype=np.float64)

    tree = build_kdtree(points)

    distances, indices = get_neighbors(
        points, tree, k
    )

    covariance, eigenvalues = calculate_covariance(
        points, indices
    )

    return {
        "distances": distances,
        "indices": indices,
        "covariance": covariance,
        "eigenvalues": eigenvalues
    }
