
import numpy as np

from features.covariance import (
    build_kdtree,
    get_neighbors,
    calculate_covariance
)


def calculate_geometric_descriptors(eigenvalues):
    """
    Calculate linearity, planarity,
    sphericity and curvature.
    """

    eigenvalues = np.maximum(eigenvalues, 0.0)

    l1 = eigenvalues[:, 0]
    l2 = eigenvalues[:, 1]
    l3 = eigenvalues[:, 2]

    epsilon = 1e-12

    # Avoid division by zero
    denominator = np.maximum(l1, epsilon)

    total = np.maximum(
        l1 + l2 + l3,
        epsilon
    )

    # Geometric descriptors
    linearity = (l1 - l2) / denominator

    planarity = (l2 - l3) / denominator

    sphericity = l3 / denominator

    curvature = l3 / total

    # Combine descriptors
    descriptors = np.column_stack([
        linearity,
        planarity,
        sphericity,
        curvature
    ])

    return descriptors


def extract_multiscale_features(
    points,
    scales=(10, 20, 40)
):
    """
    Extract geometric features at multiple
    neighbourhood scales.

    Parameters:
        points: Nx3 point coordinates
        scales: Neighbourhood sizes

    Returns:
        feature_matrix: Nx(4 * number_of_scales)
    """

    points = np.asarray(points, dtype=np.float64)

    if len(points) < 3:
        raise ValueError(
            "Insufficient points for feature extraction."
        )

    if not np.isfinite(points).all():
        raise ValueError(
            "Point cloud contains invalid coordinates."
        )

    # Build KDTree only once
    tree = build_kdtree(points)

    all_features = []

    for k in scales:

        print(f"Processing neighbourhood scale: {k}")

        # Find neighbours
        distances, indices = get_neighbors(
            points,
            tree,
            k
        )

        # Calculate covariance and eigenvalues
        covariance, eigenvalues = calculate_covariance(
            points,
            indices
        )

        # Calculate geometric descriptors
        descriptors = calculate_geometric_descriptors(
            eigenvalues
        )

        all_features.append(descriptors)

        print(
            f"Scale {k}: Feature shape = "
            f"{descriptors.shape}"
        )

    # Concatenate all scales
    feature_matrix = np.concatenate(
        all_features,
        axis=1
    )

    # Handle any numerical issues
    feature_matrix = np.nan_to_num(
        feature_matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return feature_matrix


def save_features(features, output_path):

    np.save(output_path, features)

    print("Features saved:", output_path)
    print("Feature matrix shape:", features.shape)
