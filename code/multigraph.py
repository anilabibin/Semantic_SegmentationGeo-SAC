
# ============================================================
# multigraph.py
# Geometric, Topological and Contextual Graph Construction
# ============================================================

import os
import numpy as np

from scipy.sparse import coo_matrix, save_npz
from sklearn.preprocessing import normalize


# ------------------------------------------------------------
# 1. GEOMETRIC GRAPH
# ------------------------------------------------------------

def construct_geometric_graph(
    edges,
    node_features,
    sigma_g=1.0
):
    """
    Geometric graph based on primitive feature similarity.

    Wg(i,j) = exp(-||Fi-Fj||^2 / (2*sigma_g^2))
    """

    if sigma_g <= 0:
        raise ValueError("sigma_g must be positive.")

    i = edges[:, 0]
    j = edges[:, 1]

    feature_diff = (
        node_features[i] - node_features[j]
    )

    squared_distance = np.sum(
        feature_diff ** 2, axis=1
    )

    weights = np.exp(
        -squared_distance /
        (2 * sigma_g ** 2)
    )

    return weights


# ------------------------------------------------------------
# 2. TOPOLOGICAL GRAPH
# ------------------------------------------------------------

def construct_topological_graph(
    edges,
    centroids,
    sigma_t=1.0
):
    """
    Topological graph based on spatial proximity.

    Wt(i,j) = exp(-||Ci-Cj||^2 / (2*sigma_t^2))
    """

    if sigma_t <= 0:
        raise ValueError("sigma_t must be positive.")

    i = edges[:, 0]
    j = edges[:, 1]

    spatial_diff = (
        centroids[i] - centroids[j]
    )

    squared_distance = np.sum(
        spatial_diff ** 2, axis=1
    )

    weights = np.exp(
        -squared_distance /
        (2 * sigma_t ** 2)
    )

    return weights


# ------------------------------------------------------------
# 3. CONTEXT GRAPH
# ------------------------------------------------------------

def construct_context_graph(
    edges,
    context_features
):
    """
    Context graph based on cosine similarity.

    Wc(i,j) = cosine(Fi,Fj)

    Negative cosine similarities are clipped to zero
    because this implementation uses nonnegative
    graph edge weights.
    """

    i = edges[:, 0]
    j = edges[:, 1]

    normalized_features = normalize(
        context_features,
        norm="l2",
        axis=1
    )

    similarity = np.sum(
        normalized_features[i] *
        normalized_features[j],
        axis=1
    )

    weights = np.clip(
        similarity, 0.0, 1.0
    )

    return weights


# ------------------------------------------------------------
# 4. CONVERT EDGE LIST INTO SPARSE ADJACENCY MATRIX
# ------------------------------------------------------------

def build_sparse_adjacency(
    edges,
    weights,
    num_nodes,
    self_loops=True
):
    """
    Construct a symmetric sparse adjacency matrix.
    """

    if len(edges) != len(weights):
        raise ValueError(
            "Edges and weights must have equal lengths."
        )

    if len(edges) > 0:
        if edges.min() < 0 or edges.max() >= num_nodes:
            raise ValueError("Invalid graph node index.")

    if not np.isfinite(weights).all():
        raise ValueError("Graph weights contain invalid values.")

    rows = edges[:, 0]
    cols = edges[:, 1]

    # Make the graph undirected
    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    all_weights = np.concatenate([weights, weights])

    adjacency = coo_matrix(
        (all_weights, (all_rows, all_cols)),
        shape=(num_nodes, num_nodes)
    ).tocsr()

    adjacency.sum_duplicates()

    if self_loops:
        adjacency.setdiag(1.0)

    adjacency.eliminate_zeros()

    return adjacency


# ------------------------------------------------------------
# 5. MULTI-GRAPH FUSION
# ------------------------------------------------------------

def fuse_graphs(
    A_geometry,
    A_topology,
    A_context,
    alpha=1/3,
    beta=1/3,
    gamma=1/3
):
    """
    Fuse the three sparse graphs.

    Af = alpha*Ag + beta*At + gamma*Ac
    """

    if not np.isclose(alpha + beta + gamma, 1.0):
        raise ValueError(
            "alpha + beta + gamma must equal 1."
        )

    fused = (
        alpha * A_geometry +
        beta * A_topology +
        gamma * A_context
    )

    fused = fused.tocsr()

    fused.eliminate_zeros()

    return fused


# ------------------------------------------------------------
# 6. COMPLETE MULTI-GRAPH PIPELINE
# ------------------------------------------------------------

def multigraph_pipeline(
    input_file="boundary_results/boundary_refined_graph.npz",
    output_dir="multigraph_results",

    sigma_g=1.0,
    sigma_t=1.0,

    alpha=1/3,
    beta=1/3,
    gamma=1/3,

    context_features=None
):

    os.makedirs(output_dir, exist_ok=True)

    print("\n--- LOADING REFINED GRAPH ---")

    data = np.load(input_file)

    edges = data["edges"]
    centroids = data["centroids"]
    node_features = data["node_features"]
    refined_weights = data["refined_weights"]

    num_nodes = len(centroids)

    print("Superpoints:", num_nodes)
    print("Graph edges:", len(edges))

    # Optional learned contextual representation.
    if context_features is None:
        if "context_features" in data.files:
            context_features = data["context_features"]
        else:
            print(
                "No separate context features found. "
                "Using primitive features as initial context."
            )
            context_features = node_features

    context_features = np.asarray(
        context_features, dtype=np.float64
    )

    if context_features.ndim != 2:
        raise ValueError(
            "Context features must be a 2D array."
        )

    if context_features.shape[0] != num_nodes:
        raise ValueError(
            "Context feature count does not match superpoints."
        )

    if not np.isfinite(context_features).all():
        raise ValueError(
            "Context features contain invalid values."
        )

    # --------------------------------------------------------
    # A. GEOMETRIC GRAPH
    # --------------------------------------------------------

    print("\n--- GEOMETRIC GRAPH ---")

    geometry_weights = construct_geometric_graph(
        edges,
        node_features,
        sigma_g
    )

    # Preserve boundary refinement
    geometry_weights *= refined_weights

    A_geometry = build_sparse_adjacency(
        edges,
        geometry_weights,
        num_nodes
    )

    print("Geometric graph:", A_geometry.shape)

    # --------------------------------------------------------
    # B. TOPOLOGICAL GRAPH
    # --------------------------------------------------------

    print("\n--- TOPOLOGICAL GRAPH ---")

    topology_weights = construct_topological_graph(
        edges,
        centroids,
        sigma_t
    )

    # Preserve boundary refinement
    topology_weights *= refined_weights

    A_topology = build_sparse_adjacency(
        edges,
        topology_weights,
        num_nodes
    )

    print("Topological graph:", A_topology.shape)

    # --------------------------------------------------------
    # C. CONTEXT GRAPH
    # --------------------------------------------------------

    print("\n--- CONTEXT GRAPH ---")

    context_weights = construct_context_graph(
        edges,
        context_features
    )

    # Preserve boundary refinement
    context_weights *= refined_weights

    A_context = build_sparse_adjacency(
        edges,
        context_weights,
        num_nodes
    )

    print("Context graph:", A_context.shape)

    # --------------------------------------------------------
    # D. GRAPH FUSION
    # --------------------------------------------------------

    print("\n--- MULTI-GRAPH FUSION ---")

    A_fused = fuse_graphs(
        A_geometry,
        A_topology,
        A_context,
        alpha,
        beta,
        gamma
    )

    print("Fused graph:", A_fused.shape)

    # --------------------------------------------------------
    # E. SAVE ALL GRAPHS
    # --------------------------------------------------------

    save_npz(
        os.path.join(
            output_dir, "geometric_graph.npz"
        ),
        A_geometry
    )

    save_npz(
        os.path.join(
            output_dir, "topological_graph.npz"
        ),
        A_topology
    )

    save_npz(
        os.path.join(
            output_dir, "context_graph.npz"
        ),
        A_context
    )

    save_npz(
        os.path.join(
            output_dir, "fused_graph.npz"
        ),
        A_fused
    )

    # Save edge-level graph data for inspection
    np.savez_compressed(
        os.path.join(
            output_dir, "multigraph_edge_data.npz"
        ),

        edges=edges,

        geometry_weights=geometry_weights,
        topology_weights=topology_weights,
        context_weights=context_weights,

        refined_weights=refined_weights,

        alpha=alpha,
        beta=beta,
        gamma=gamma
    )

    print("\nAll graphs saved successfully.")

    print("Output directory:", output_dir)

    return {
        "A_geometry": A_geometry,
        "A_topology": A_topology,
        "A_context": A_context,
        "A_fused": A_fused
    }


# ------------------------------------------------------------
# 7. EXECUTION
# ------------------------------------------------------------

if __name__ == "__main__":

    graphs = multigraph_pipeline(

        input_file=(
            "boundary_results/"
            "boundary_refined_graph.npz"
        ),

        output_dir="multigraph_results",

        sigma_g=1.0,
        sigma_t=1.0,

        alpha=1/3,
        beta=1/3,
        gamma=1/3
    )

    print("\nMulti-graph construction completed.")
