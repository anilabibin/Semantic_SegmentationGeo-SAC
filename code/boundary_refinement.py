
# ============================================================
# boundary_detection_refinement.py
# Structural Boundary Detection and Boundary-Aware Graph Refinement
# ============================================================

import os
import numpy as np

from scipy.spatial import cKDTree
from sklearn.preprocessing import normalize


# ------------------------------------------------------------
# 1. SUPERPOINT FEATURE AGGREGATION
# ------------------------------------------------------------

def aggregate_superpoints(
    points,
    normals,
    primitive_features,
    superpoint_ids
):
    """
    Convert point-level features into superpoint-level features.

    Inputs:
        points             : (N, 3)
        normals            : (N, 3)
        primitive_features : (N, F)
        superpoint_ids     : (N,)

    Returns:
        centroids
        mean_normals
        node_features
        heights
        unique_ids
    """

    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    primitive_features = np.asarray(
        primitive_features, dtype=np.float64
    )
    superpoint_ids = np.asarray(superpoint_ids)

    if len(points) == 0:
        raise ValueError("Input point cloud is empty.")

    if not (
        len(points) == len(normals) ==
        len(primitive_features) == len(superpoint_ids)
    ):
        raise ValueError("Input arrays have inconsistent lengths.")

    if not np.isfinite(points).all():
        raise ValueError("Points contain NaN or infinite values.")

    if not np.isfinite(normals).all():
        raise ValueError("Normals contain NaN or infinite values.")

    if not np.isfinite(primitive_features).all():
        raise ValueError("Primitive features contain invalid values.")

    unique_ids, inverse = np.unique(
        superpoint_ids, return_inverse=True
    )

    num_nodes = len(unique_ids)

    centroids = np.zeros((num_nodes, 3))
    mean_normals = np.zeros((num_nodes, 3))
    node_features = np.zeros(
        (num_nodes, primitive_features.shape[1])
    )
    heights = np.zeros(num_nodes)

    for i in range(num_nodes):

        mask = inverse == i

        centroids[i] = points[mask].mean(axis=0)

        mean_normals[i] = normals[mask].mean(axis=0)

        node_features[i] = primitive_features[mask].mean(axis=0)

        heights[i] = points[mask, 2].mean()

    mean_normals = normalize(
        mean_normals, norm="l2"
    )

    return (
        centroids,
        mean_normals,
        node_features,
        heights,
        unique_ids
    )


# ------------------------------------------------------------
# 2. CREATE SUPERPOINT NEIGHBOUR GRAPH
# ------------------------------------------------------------

def construct_initial_graph(
    centroids,
    k=12,
    sigma=1.0
):
    """
    Construct a symmetric KNN graph.

    Returns:
        edges: (E, 2)
        weights: (E,)
    """

    n = len(centroids)

    if n < 2:
        raise ValueError(
            "At least two superpoints are required."
        )

    k = min(k, n - 1)

    tree = cKDTree(centroids)

    distances, indices = tree.query(
        centroids, k=k + 1
    )

    edge_dict = {}

    for i in range(n):

        for j, distance in zip(
            indices[i, 1:],
            distances[i, 1:]
        ):

            j = int(j)

            if i == j:
                continue

            edge = (min(i, j), max(i, j))

            # Gaussian spatial affinity
            weight = np.exp(
                -(distance ** 2) / (2 * sigma ** 2)
            )

            edge_dict[edge] = max(
                edge_dict.get(edge, 0.0),
                weight
            )

    edges = np.array(
        list(edge_dict.keys()), dtype=np.int64
    ).reshape(-1, 2)

    weights = np.array(
        list(edge_dict.values()), dtype=np.float64
    )

    return edges, weights


# ------------------------------------------------------------
# 3. ROBUST NORMALIZATION
# ------------------------------------------------------------

def robust_normalize(values):
    """
    Normalize boundary measurements using
    the 95th percentile of their magnitudes.
    """

    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return values

    scale = np.percentile(
        np.abs(values), 95
    )

    if scale < 1e-10:
        return np.zeros_like(values)

    return np.clip(
        values / scale, 0, 1
    )


# ------------------------------------------------------------
# 4. STRUCTURAL BOUNDARY DETECTION
# ------------------------------------------------------------

def detect_boundaries(
    edges,
    node_features,
    normals,
    heights,
    alpha=0.4,
    beta=0.3,
    gamma=0.3
):
    """
    Calculate structural boundary strength.

    Dg: Geometric feature difference
    Dn: Normal orientation difference
    Dh: Height difference

    B = alpha*Dg + beta*Dn + gamma*Dh
    """

    if not np.isclose(alpha + beta + gamma, 1.0):
        raise ValueError(
            "alpha + beta + gamma must equal 1."
        )

    i = edges[:, 0]
    j = edges[:, 1]

    # Geometric boundary cue
    geometry_diff = np.linalg.norm(
        node_features[i] - node_features[j],
        axis=1
    )

    # Normal boundary cue
    normal_similarity = np.sum(
        normals[i] * normals[j], axis=1
    )

    normal_diff = 1.0 - np.clip(
        np.abs(normal_similarity), 0.0, 1.0
    )

    # Height boundary cue
    height_diff = np.abs(
        heights[i] - heights[j]
    )

    # Normalize each cue
    geometry_diff = robust_normalize(
        geometry_diff
    )

    normal_diff = np.clip(
        normal_diff, 0, 1
    )

    height_diff = robust_normalize(
        height_diff
    )

    # Combined boundary strength
    boundary_strength = (
        alpha * geometry_diff +
        beta * normal_diff +
        gamma * height_diff
    )

    return {
        "geometry_boundary": geometry_diff,
        "normal_boundary": normal_diff,
        "height_boundary": height_diff,
        "boundary_strength": boundary_strength
    }


# ------------------------------------------------------------
# 5. BOUNDARY-AWARE GRAPH REFINEMENT
# ------------------------------------------------------------

def refine_graph(
    edges,
    initial_weights,
    boundary_strength,
    tau=0.5,
    remove_threshold=None
):
    """
    Refine graph connections.

    Refined weight:
        A_new = A_initial * exp(-B/tau)

    Optional threshold can remove weak refined edges.
    """

    if tau <= 0:
        raise ValueError("tau must be positive.")

    refined_weights = (
        initial_weights *
        np.exp(-boundary_strength / tau)
    )

    if remove_threshold is not None:

        mask = refined_weights >= remove_threshold

        edges = edges[mask]

        refined_weights = refined_weights[mask]

        boundary_strength = boundary_strength[mask]

    return edges, refined_weights, boundary_strength


# ------------------------------------------------------------
# 6. COMPLETE PIPELINE
# ------------------------------------------------------------

def boundary_pipeline(
    points,
    normals,
    primitive_features,
    superpoint_ids,
    output_dir="boundary_results",
    k=12,
    sigma=1.0,
    alpha=0.4,
    beta=0.3,
    gamma=0.3,
    tau=0.5,
    remove_threshold=None
):

    os.makedirs(output_dir, exist_ok=True)

    print("\n--- SUPERPOINT AGGREGATION ---")

    (
        centroids,
        mean_normals,
        node_features,
        heights,
        unique_ids
    ) = aggregate_superpoints(
        points,
        normals,
        primitive_features,
        superpoint_ids
    )

    print("Number of superpoints:", len(centroids))

    print("\n--- INITIAL GRAPH CONSTRUCTION ---")

    edges, initial_weights = construct_initial_graph(
        centroids, k=k, sigma=sigma
    )

    print("Initial graph edges:", len(edges))

    print("\n--- STRUCTURAL BOUNDARY DETECTION ---")

    boundary_data = detect_boundaries(
        edges,
        node_features,
        mean_normals,
        heights,
        alpha,
        beta,
        gamma
    )

    boundary_strength = boundary_data[
        "boundary_strength"
    ]

    print(
        "Mean boundary strength:",
        np.mean(boundary_strength)
    )

    print("\n--- BOUNDARY-AWARE GRAPH REFINEMENT ---")

    refined_edges, refined_weights, refined_boundary = (
        refine_graph(
            edges,
            initial_weights,
            boundary_strength,
            tau,
            remove_threshold
        )
    )

    print("Refined graph edges:", len(refined_edges))

    # Save all outputs
    output_file = os.path.join(
        output_dir,
        "boundary_refined_graph.npz"
    )

    np.savez_compressed(
        output_file,

        centroids=centroids,
        normals=mean_normals,
        node_features=node_features,
        heights=heights,
        unique_ids=unique_ids,

        edges=refined_edges,
        initial_weights=initial_weights
            if remove_threshold is None
            else np.empty(0),

        refined_weights=refined_weights,
        boundary_strength=refined_boundary
    )

    # Save individual boundary measurements
    boundary_file = os.path.join(
        output_dir,
        "boundary_measurements.npz"
    )

    np.savez_compressed(
        boundary_file,
        edges=edges,
        geometry_boundary=boundary_data[
            "geometry_boundary"
        ],
        normal_boundary=boundary_data[
            "normal_boundary"
        ],
        height_boundary=boundary_data[
            "height_boundary"
        ],
        boundary_strength=boundary_data[
            "boundary_strength"
        ]
    )

    print("\nSaved refined graph:", output_file)
    print("Saved boundary measurements:", boundary_file)

    return {
        "centroids": centroids,
        "node_features": node_features,
        "edges": refined_edges,
        "weights": refined_weights,
        "boundary_strength": refined_boundary
    }


# ------------------------------------------------------------
# 7. EXAMPLE EXECUTION
# ------------------------------------------------------------

if __name__ == "__main__":

    # Replace this path with your feature file.
    data = np.load("superpoint_input.npz")

    points = data["points"]
    normals = data["normals"]
    primitive_features = data["primitive_features"]
    superpoint_ids = data["superpoint_ids"]

    results = boundary_pipeline(
        points=points,
        normals=normals,
        primitive_features=primitive_features,
        superpoint_ids=superpoint_ids,

        output_dir="boundary_results",

        k=12,
        sigma=1.0,

        alpha=0.4,
        beta=0.3,
        gamma=0.3,

        tau=0.5,

        remove_threshold=None
    )

    print("\nBoundary detection and refinement completed.")
