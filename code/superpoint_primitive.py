"""
Combined superpoint generation and primitive grouping for 3D point clouds.

Pipeline:
1. Load LAS/LAZ, PLY/PCD, or XYZ/TXT/CSV point cloud.
2. Estimate normals.
3. Compute local covariance eigenvalue descriptors at multiple KNN scales.
4. Build a KNN graph and retain edges with compatible geometry.
5. Extract connected components as superpoints.
6. Describe each superpoint and assign a geometric primitive type.
7. Group adjacent, geometrically similar superpoints with the same primitive type.
8. Save point-level IDs and primitive labels to CSV, NPZ, and (for LAS/LAZ input) a LAS output.

This is a geometry-driven, non-learning implementation. It does not predict semantic classes
such as building/tree/road; primitive types are geometric categories:
linear, planar, volumetric, or irregular.

Dependencies:
    pip install numpy scipy scikit-learn open3d laspy lazrs

Example:
    python superpoint_primitive.py --input "D:\\data\\scene.las" --output_dir "D:\\data\\results"

For a text file, the first three columns must be X,Y,Z:
    python superpoint_primitive.py --input "D:\\data\\scene.txt" --output_dir "D:\\data\\results"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


# ----------------------------- CONFIGURATION -----------------------------

# Descriptor neighbourhood sizes. The actual value is clipped to point count.
DEFAULT_SCALES = (12, 24, 40)

# Number of neighbours used to construct the point graph.
DEFAULT_GRAPH_K = 12

# Edge compatibility thresholds. Tune against your point density and scene.
DEFAULT_NORMAL_COS = 0.82
DEFAULT_FEATURE_DISTANCE = 0.45

# Remove very small connected components by merging them into the nearest
# larger component. Set to 1 to disable size-based merging.
DEFAULT_MIN_SUPERPOINT_SIZE = 20

# Primitive grouping thresholds.
DEFAULT_PRIMITIVE_SIMILARITY = 0.70


# ----------------------------- INPUT / OUTPUT -----------------------------

def load_point_cloud(path: Path):
    """Return XYZ, optional RGB, and input metadata."""
    suffix = path.suffix.lower()
    rgb = None
    metadata = {"format": suffix}

    if suffix in {".las", ".laz"}:
        try:
            import laspy
        except ImportError as exc:
            raise ImportError("Install laspy and lazrs: pip install laspy lazrs") from exc

        las = laspy.read(str(path))
        xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
        if all(hasattr(las, c) for c in ("red", "green", "blue")):
            rgb = np.column_stack((las.red, las.green, las.blue))
        metadata["las_header"] = las.header
        metadata["las_scales"] = tuple(las.header.scales)
        metadata["las_offsets"] = tuple(las.header.offsets)
        metadata["las_point_format"] = las.header.point_format.id
        metadata["las_version"] = str(las.header.version)
        metadata["las"] = las
    elif suffix in {".ply", ".pcd"}:
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError("Install Open3D: pip install open3d") from exc

        pcd = o3d.io.read_point_cloud(str(path))
        xyz = np.asarray(pcd.points, dtype=np.float64)
        if pcd.has_colors():
            rgb = np.asarray(pcd.colors, dtype=np.float64)
        metadata["open3d"] = True
    elif suffix in {".txt", ".xyz", ".csv"}:
        delimiter = "," if suffix == ".csv" else None
        arr = np.loadtxt(str(path), delimiter=delimiter, comments="#")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 3:
            raise ValueError("Text input must contain at least three columns: X Y Z.")
        xyz = arr[:, :3].astype(np.float64)
        if arr.shape[1] >= 6:
            rgb = arr[:, 3:6].copy()
        metadata["text_extra_columns"] = int(arr.shape[1] - 3)
    else:
        raise ValueError(f"Unsupported input format: {suffix}")

    if xyz.size == 0:
        raise ValueError("The input point cloud contains zero points.")
    valid = np.isfinite(xyz).all(axis=1)
    if not np.all(valid):
        print(f"[WARN] Removing {(~valid).sum()} points with invalid XYZ values.")
        xyz = xyz[valid]
        if rgb is not None:
            rgb = rgb[valid]
        metadata["valid_mask"] = valid

    if len(xyz) < 4:
        raise ValueError("At least four valid points are required.")

    return xyz, rgb, metadata


def estimate_normals(xyz: np.ndarray, k: int = 20) -> np.ndarray:
    """Estimate unoriented normals from local covariance eigenvectors."""
    n = len(xyz)
    k = min(max(4, int(k)), n)
    tree = cKDTree(xyz)
    _, nn = tree.query(xyz, k=k)
    normals = np.zeros((n, 3), dtype=np.float64)

    # Batch covariance calculation; loop avoids a huge N x K x 3 temporary.
    for start in range(0, n, 4096):
        end = min(start + 4096, n)
        neigh = xyz[nn[start:end]]
        centered = neigh - neigh.mean(axis=1, keepdims=True)
        cov = np.einsum("bki,bkj->bij", centered, centered) / max(k - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[start:end] = eigvecs[:, :, 0]

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(lengths, 1e-12)
    return normals


# ----------------------------- GEOMETRIC FEATURES -----------------------------

def compute_multiscale_descriptors(
    xyz: np.ndarray,
    scales: Tuple[int, ...] = DEFAULT_SCALES,
):
    """
    Compute [linearity, planarity, sphericity, curvature] at each KNN scale.
    Returns concatenated descriptors and descriptors from the largest scale.
    """
    n = len(xyz)
    tree = cKDTree(xyz)
    all_scales = []

    for requested_k in scales:
        k = min(max(4, int(requested_k)), n)
        _, nn = tree.query(xyz, k=k)
        scale_features = np.zeros((n, 4), dtype=np.float32)

        for start in range(0, n, 4096):
            end = min(start + 4096, n)
            neigh = xyz[nn[start:end]]
            centered = neigh - neigh.mean(axis=1, keepdims=True)
            cov = np.einsum("bki,bkj->bij", centered, centered) / max(k - 1, 1)
            eig = np.linalg.eigvalsh(cov)
            eig = np.maximum(eig[:, ::-1], 1e-12)  # descending: l1,l2,l3
            l1, l2, l3 = eig[:, 0], eig[:, 1], eig[:, 2]
            denom = np.maximum(l1, 1e-12)

            linearity = (l1 - l2) / denom
            planarity = (l2 - l3) / denom
            sphericity = l3 / denom
            curvature = l3 / np.maximum(l1 + l2 + l3, 1e-12)

            scale_features[start:end] = np.column_stack(
                (linearity, planarity, sphericity, curvature)
            )

        all_scales.append(scale_features)

    concatenated = np.concatenate(all_scales, axis=1)
    largest_scale = all_scales[-1]
    return concatenated, largest_scale


def robust_normalize_features(features: np.ndarray) -> np.ndarray:
    """Robustly scale each descriptor dimension using its 5th/95th percentiles."""
    lo = np.percentile(features, 5, axis=0)
    hi = np.percentile(features, 95, axis=0)
    scaled = (features - lo) / np.maximum(hi - lo, 1e-8)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


# ----------------------------- SUPERPOINT GENERATION -----------------------------

def build_superpoint_graph(
    xyz: np.ndarray,
    normals: np.ndarray,
    normalized_features: np.ndarray,
    graph_k: int = DEFAULT_GRAPH_K,
    normal_cos: float = DEFAULT_NORMAL_COS,
    feature_distance: float = DEFAULT_FEATURE_DISTANCE,
):
    """
    Construct an undirected KNN graph.
    Neighbour edges are retained when normal orientation and primitive
    descriptors are sufficiently compatible. Spatial KNN defines locality.
    """
    n = len(xyz)
    k = min(max(2, graph_k + 1), n)
    tree = cKDTree(xyz)
    distances, indices = tree.query(xyz, k=k)

    rows, cols = [], []
    for i in range(n):
        for pos in range(1, k):  # skip self-neighbour
            j = int(indices[i, pos])
            if i == j:
                continue

            # Normals may have opposite signs, so use absolute dot product.
            normal_similarity = abs(float(np.dot(normals[i], normals[j])))
            feature_dist = float(
                np.linalg.norm(normalized_features[i] - normalized_features[j])
                / np.sqrt(normalized_features.shape[1])
            )

            if (
                normal_similarity >= normal_cos
                and feature_dist <= feature_distance
            ):
                rows.extend((i, j))
                cols.extend((j, i))

    if not rows:
        # A graph with isolated vertices is valid, but warn the user.
        print("[WARN] No compatible KNN edges. Every point may become its own superpoint.")
        return coo_matrix((n, n), dtype=np.uint8).tocsr()

    data = np.ones(len(rows), dtype=np.uint8)
    adjacency = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    adjacency.data[:] = 1
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency


def relabel_components(labels: np.ndarray) -> np.ndarray:
    """Relabel component IDs consecutively from zero."""
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.astype(np.int32)


def merge_tiny_superpoints(
    xyz: np.ndarray,
    superpoint_ids: np.ndarray,
    min_size: int,
    k: int = 8,
) -> np.ndarray:
    """
    Merge small components into the most frequent neighbouring component
    among their K nearest points. This is a practical post-processing step.
    """
    if min_size <= 1:
        return superpoint_ids

    ids = superpoint_ids.copy()
    counts = np.bincount(ids)
    small_ids = np.where(counts < min_size)[0]
    if len(small_ids) == 0:
        return ids

    tree = cKDTree(xyz)
    _, nn = tree.query(xyz, k=min(k + 1, len(xyz)))

    for sid in small_ids:
        point_indices = np.where(ids == sid)[0]
        neighbour_ids = ids[nn[point_indices, 1:]].ravel()
        neighbour_ids = neighbour_ids[neighbour_ids != sid]
        if neighbour_ids.size == 0:
            continue
        candidate_ids, candidate_counts = np.unique(neighbour_ids, return_counts=True)
        ids[point_indices] = candidate_ids[np.argmax(candidate_counts)]

    return relabel_components(ids)


def generate_superpoints(
    xyz: np.ndarray,
    normals: np.ndarray,
    normalized_features: np.ndarray,
    graph_k: int,
    normal_cos: float,
    feature_distance: float,
    min_size: int,
):
    adjacency = build_superpoint_graph(
        xyz,
        normals,
        normalized_features,
        graph_k=graph_k,
        normal_cos=normal_cos,
        feature_distance=feature_distance,
    )
    n_components, labels = connected_components(
        adjacency, directed=False, return_labels=True
    )
    labels = relabel_components(labels)
    print(f"[INFO] Initial connected components: {n_components}")

    labels = merge_tiny_superpoints(xyz, labels, min_size=min_size)
    print(f"[INFO] Superpoints after small-region merging: {len(np.unique(labels))}")
    return labels, adjacency


# ----------------------------- PRIMITIVE GROUPING -----------------------------

PRIMITIVE_NAMES = {
    0: "linear",
    1: "planar",
    2: "volumetric",
    3: "irregular",
}


def describe_superpoints(
    xyz: np.ndarray,
    normals: np.ndarray,
    point_features: np.ndarray,
    superpoint_ids: np.ndarray,
):
    """Calculate one descriptor row and centroid for every superpoint."""
    unique_ids = np.unique(superpoint_ids)
    records = []

    for sid in unique_ids:
        idx = np.where(superpoint_ids == sid)[0]
        pts = xyz[idx]
        feat = point_features[idx]
        nrm = normals[idx]

        centroid = pts.mean(axis=0)
        extent = pts.max(axis=0) - pts.min(axis=0)
        mean_feat = feat.mean(axis=0)
        mean_normal = nrm.mean(axis=0)
        mean_normal /= max(np.linalg.norm(mean_normal), 1e-12)

        # Use largest-scale descriptors: linearity, planarity, sphericity, curvature.
        lin, pla, sph, curv = mean_feat

        # Select the dominant primitive descriptor; mark ambiguous/high-curvature
        # regions as irregular.
        dominant = int(np.argmax([lin, pla, sph]))
        if curv > 0.25 or max(lin, pla, sph) < 0.20:
            primitive_id = 3
        else:
            primitive_id = dominant

        records.append({
            "superpoint_id": int(sid),
            "point_count": int(len(idx)),
            "centroid": centroid,
            "extent": extent,
            "mean_normal": mean_normal,
            "features": mean_feat,
            "primitive_id": primitive_id,
            "primitive_name": PRIMITIVE_NAMES[primitive_id],
        })

    return records


def group_primitives(
    xyz: np.ndarray,
    superpoint_ids: np.ndarray,
    records,
    similarity_threshold: float = DEFAULT_PRIMITIVE_SIMILARITY,
):
    """
    Group adjacent superpoints when they have the same primitive type and
    similar normalized geometric descriptors. Adjacency is inferred from
    neighbouring point pairs in the original point cloud.
    """
    unique_ids = np.unique(superpoint_ids)
    n_sp = len(unique_ids)
    id_to_pos = {int(sid): pos for pos, sid in enumerate(unique_ids)}
    record_by_id = {r["superpoint_id"]: r for r in records}

    # Build a compact descriptor per superpoint.
    sp_features = np.vstack([
        record_by_id[int(sid)]["features"] for sid in unique_ids
    ])
    sp_types = np.array([
        record_by_id[int(sid)]["primitive_id"] for sid in unique_ids
    ], dtype=np.int32)

    lo = sp_features.min(axis=0)
    hi = sp_features.max(axis=0)
    sp_norm = (sp_features - lo) / np.maximum(hi - lo, 1e-8)

    # Find neighbouring superpoints through KNN point pairs.
    tree = cKDTree(xyz)
    _, nn = tree.query(xyz, k=min(9, len(xyz)))
    candidate_pairs = set()

    for i in range(len(xyz)):
        sid_i = int(superpoint_ids[i])
        for j in nn[i, 1:]:
            sid_j = int(superpoint_ids[int(j)])
            if sid_i != sid_j:
                a, b = sorted((sid_i, sid_j))
                candidate_pairs.add((a, b))

    parent = np.arange(n_sp, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for sid_a, sid_b in candidate_pairs:
        pa, pb = id_to_pos[sid_a], id_to_pos[sid_b]

        if sp_types[pa] != sp_types[pb]:
            continue

        similarity = 1.0 - float(
            np.linalg.norm(sp_norm[pa] - sp_norm[pb])
            / np.sqrt(sp_norm.shape[1])
        )

        if similarity >= similarity_threshold:
            union(pa, pb)

    roots = np.array([find(i) for i in range(n_sp)])
    group_for_position = relabel_components(roots)
    group_for_superpoint = {
        int(sid): int(group_for_position[id_to_pos[int(sid)]])
        for sid in unique_ids
    }

    point_group_ids = np.array(
        [group_for_superpoint[int(sid)] for sid in superpoint_ids],
        dtype=np.int32,
    )

    point_primitive_ids = np.array(
        [record_by_id[int(sid)]["primitive_id"] for sid in superpoint_ids],
        dtype=np.uint8,
    )

    return point_group_ids, point_primitive_ids, group_for_superpoint


# ----------------------------- SAVING RESULTS -----------------------------

def save_outputs(
    input_path: Path,
    output_dir: Path,
    xyz: np.ndarray,
    rgb: Optional[np.ndarray],
    metadata: dict,
    superpoint_ids: np.ndarray,
    primitive_group_ids: np.ndarray,
    primitive_ids: np.ndarray,
    records,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    csv_path = output_dir / f"{stem}_point_labels.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "x", "y", "z", "superpoint_id",
            "primitive_group_id", "primitive_id", "primitive_name"
        ])
        for i in range(len(xyz)):
            writer.writerow([
                xyz[i, 0], xyz[i, 1], xyz[i, 2],
                int(superpoint_ids[i]),
                int(primitive_group_ids[i]),
                int(primitive_ids[i]),
                PRIMITIVE_NAMES[int(primitive_ids[i])],
            ])

    npz_path = output_dir / f"{stem}_superpoints.npz"
    np.savez_compressed(
        npz_path,
        xyz=xyz,
        superpoint_id=superpoint_ids,
        primitive_group_id=primitive_group_ids,
        primitive_id=primitive_ids,
        rgb=rgb if rgb is not None else np.empty((0, 3)),
    )

    summary_path = output_dir / f"{stem}_primitive_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "superpoint_id", "point_count", "primitive_id",
            "primitive_name", "centroid_x", "centroid_y", "centroid_z",
            "extent_x", "extent_y", "extent_z",
            "linearity", "planarity", "sphericity", "curvature",
        ])
        for r in records:
            c = r["centroid"]
            e = r["extent"]
            feat = r["features"]
            writer.writerow([
                r["superpoint_id"], r["point_count"],
                r["primitive_id"], r["primitive_name"],
                *c, *e, *feat,
            ])

    # Write an additional LAS file when input is LAS/LAZ.
    las_path = None
    if input_path.suffix.lower() in {".las", ".laz"}:
        try:
            import laspy
            source_las = metadata["las"]
            header = source_las.header.copy()
            out_las = laspy.LasData(header)
            valid_mask = metadata.get("valid_mask")
            if valid_mask is not None:
                out_las.points = source_las.points[valid_mask].copy()
            else:
                out_las.points = source_las.points.copy()

            # Custom Extra Bytes avoid misusing standard ASPRS classification codes.
            existing = set(out_las.point_format.dimension_names)
            if "superpoint_id" not in existing:
                out_las.add_extra_dim(
                    laspy.ExtraBytesParams(
                        name="superpoint_id", type=np.uint32,
                        description="Geometry-derived superpoint ID"
                    )
                )
            if "primitive_group_id" not in existing:
                out_las.add_extra_dim(
                    laspy.ExtraBytesParams(
                        name="primitive_group_id", type=np.uint32,
                        description="Geometry-derived primitive group ID"
                    )
                )
            if "primitive_type" not in existing:
                out_las.add_extra_dim(
                    laspy.ExtraBytesParams(
                        name="primitive_type", type=np.uint8,
                        description="0 linear, 1 planar, 2 volumetric, 3 irregular"
                    )
                )

            out_las.x = xyz[:, 0]
            out_las.y = xyz[:, 1]
            out_las.z = xyz[:, 2]
            out_las["superpoint_id"] = superpoint_ids.astype(np.uint32)
            out_las["primitive_group_id"] = primitive_group_ids.astype(np.uint32)
            out_las["primitive_type"] = primitive_ids.astype(np.uint8)

            las_path = output_dir / f"{stem}_superpoints_primitives.las"
            out_las.write(str(las_path))
        except Exception as exc:
            print(f"[WARN] Could not write LAS output: {exc}")

    config_path = output_dir / f"{stem}_run_info.json"
    config = {
        "input": str(input_path),
        "number_of_points": int(len(xyz)),
        "number_of_superpoints": int(len(np.unique(superpoint_ids))),
        "number_of_primitive_groups": int(len(np.unique(primitive_group_ids))),
        "primitive_types": PRIMITIVE_NAMES,
        "note": "Primitive labels are geometric descriptors, not semantic object classes.",
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\n[SAVED]")
    print("Point labels CSV:", csv_path)
    print("Superpoint NPZ:", npz_path)
    print("Superpoint summary CSV:", summary_path)
    if las_path:
        print("LAS output:", las_path)
    print("Run information:", config_path)


# ----------------------------- MAIN -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Geometry-driven superpoint generation and primitive grouping."
    )
    parser.add_argument("--input", required=True, help="Input LAS/LAZ, PLY/PCD, TXT/XYZ/CSV.")
    parser.add_argument("--output_dir", required=True, help="Directory for outputs.")
    parser.add_argument("--scales", nargs="+", type=int, default=list(DEFAULT_SCALES))
    parser.add_argument("--normal_k", type=int, default=20)
    parser.add_argument("--graph_k", type=int, default=DEFAULT_GRAPH_K)
    parser.add_argument("--normal_cos", type=float, default=DEFAULT_NORMAL_COS)
    parser.add_argument("--feature_distance", type=float, default=DEFAULT_FEATURE_DISTANCE)
    parser.add_argument("--min_superpoint_size", type=int, default=DEFAULT_MIN_SUPERPOINT_SIZE)
    parser.add_argument("--primitive_similarity", type=float, default=DEFAULT_PRIMITIVE_SIMILARITY)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    print("[1/7] Loading point cloud...")
    xyz, rgb, metadata = load_point_cloud(input_path)
    print(f"[INFO] Valid input points: {len(xyz):,}")

    print("[2/7] Estimating normals...")
    normals = estimate_normals(xyz, k=args.normal_k)

    print("[3/7] Computing multi-scale primitive descriptors...")
    _, largest_scale_features = compute_multiscale_descriptors(
        xyz, scales=tuple(args.scales)
    )
    normalized_features = robust_normalize_features(largest_scale_features)

    print("[4/7] Generating superpoints...")
    superpoint_ids, _ = generate_superpoints(
        xyz=xyz,
        normals=normals,
        normalized_features=normalized_features,
        graph_k=args.graph_k,
        normal_cos=args.normal_cos,
        feature_distance=args.feature_distance,
        min_size=args.min_superpoint_size,
    )

    print("[5/7] Describing superpoints and assigning primitive types...")
    records = describe_superpoints(
        xyz, normals, largest_scale_features, superpoint_ids
    )

    print("[6/7] Grouping geometrically similar primitives...")
    primitive_group_ids, primitive_ids, _ = group_primitives(
        xyz,
        superpoint_ids,
        records,
        similarity_threshold=args.primitive_similarity,
    )

    print("[7/7] Saving outputs...")
    save_outputs(
        input_path=input_path,
        output_dir=output_dir,
        xyz=xyz,
        rgb=rgb,
        metadata=metadata,
        superpoint_ids=superpoint_ids,
        primitive_group_ids=primitive_group_ids,
        primitive_ids=primitive_ids,
        records=records,
    )

    print("\n[DONE]")
    print(f"Points: {len(xyz):,}")
    print(f"Superpoints: {len(np.unique(superpoint_ids)):,}")
    print(f"Primitive groups: {len(np.unique(primitive_group_ids)):,}")
    for pid, name in PRIMITIVE_NAMES.items():
        count = int(np.sum(primitive_ids == pid))
        print(f"{name}: {count:,} points")


if __name__ == "__main__":
    main()
