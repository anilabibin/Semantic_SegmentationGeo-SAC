
import os
import numpy as np
import open3d as o3d
import laspy


# --------------------------------------------------
# 1. SEMANTIC CLASS COLOURS
# --------------------------------------------------

CLASS_NAMES = {
    0: "Building",
    1: "Tree",
    2: "Road",
    3: "Car",
    4: "Terrain"
}

CLASS_COLORS = {
    0: [1.0, 0.0, 0.0],   # Red
    1: [0.0, 0.8, 0.0],   # Green
    2: [0.5, 0.5, 0.5],   # Grey
    3: [0.0, 0.3, 1.0],   # Blue
    4: [0.8, 0.6, 0.2]    # Brown
}


# --------------------------------------------------
# 2. GENERATE POINT COLOURS
# --------------------------------------------------

def get_semantic_colors(labels):

    labels = np.asarray(labels).astype(int)

    colors = np.zeros(
        (len(labels), 3),
        dtype=np.float64
    )

    for class_id, color in CLASS_COLORS.items():

        mask = labels == class_id

        colors[mask] = color

    return colors


# --------------------------------------------------
# 3. CREATE POINT CLOUD
# --------------------------------------------------

def create_point_cloud(points, labels=None):

    points = np.asarray(points)

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        points[:, :3]
    )

    if labels is not None:

        colors = get_semantic_colors(labels)

        pcd.colors = o3d.utility.Vector3dVector(
            colors
        )

    elif points.shape[1] >= 6:

        rgb = points[:, 3:6].astype(float)

        if rgb.max() > 1:
            rgb = rgb / 255.0

        pcd.colors = o3d.utility.Vector3dVector(
            np.clip(rgb, 0, 1)
        )

    return pcd


# --------------------------------------------------
# 4. VISUALIZE POINT CLOUD
# --------------------------------------------------

def visualize_point_cloud(
    points,
    labels=None,
    window_name="Semantic Point Cloud"
):

    pcd = create_point_cloud(
        points,
        labels
    )

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=window_name,
        width=1200,
        height=800
    )


# --------------------------------------------------
# 5. COMPARE GROUND TRUTH AND PREDICTION
# --------------------------------------------------

def compare_segmentation(
    points,
    ground_truth,
    predictions
):

    gt_cloud = create_point_cloud(
        points,
        ground_truth
    )

    pred_cloud = create_point_cloud(
        points,
        predictions
    )

    gt_cloud.translate([-10, 0, 0])

    pred_cloud.translate([10, 0, 0])

    print("Left: Ground Truth")
    print("Right: Predicted Segmentation")

    o3d.visualization.draw_geometries(
        [gt_cloud, pred_cloud],
        window_name="Ground Truth vs Prediction",
        width=1400,
        height=800
    )


# --------------------------------------------------
# 6. SAVE SEMANTIC POINT CLOUD AS PLY
# --------------------------------------------------

def save_ply(
    points,
    labels,
    output_path
):

    pcd = create_point_cloud(
        points,
        labels
    )

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    o3d.io.write_point_cloud(
        output_path,
        pcd
    )

    print("PLY saved:", output_path)


# --------------------------------------------------
# 7. SAVE SEMANTIC POINT CLOUD AS LAS
# --------------------------------------------------

def save_las(
    points,
    labels,
    output_path
):

    points = np.asarray(points)

    labels = np.asarray(labels).astype(np.uint8)

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    if len(points) != len(labels):
        raise ValueError(
            "Point count and label count must match."
        )

    header = laspy.LasHeader(
        point_format=3,
        version="1.2"
    )

    # Preserve coordinate precision using scales
    scales = np.array([0.001, 0.001, 0.001])

    header.scales = scales

    header.offsets = np.floor(
        points[:, :3].min(axis=0)
    )

    las = laspy.LasData(header)

    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    # These are custom semantic IDs, not standard
    # ASPRS LAS class meanings.
    las.classification = labels

    colors = (
        get_semantic_colors(labels) * 65535
    ).astype(np.uint16)

    las.red = colors[:, 0]
    las.green = colors[:, 1]
    las.blue = colors[:, 2]

    las.write(output_path)

    print("LAS saved:", output_path)


# --------------------------------------------------
# 8. LOAD AND VISUALIZE EVALUATION RESULT
# --------------------------------------------------

def visualize_saved_result(npz_file):

    data = np.load(npz_file)

    points = data["points"]

    predictions = data["predictions"]

    visualize_point_cloud(
        points,
        predictions,
        window_name="Predicted Semantic Segmentation"
    )

    if "targets" in data:

        compare_segmentation(
            points,
            data["targets"],
            predictions
        )


# --------------------------------------------------
# 9. MAIN
# --------------------------------------------------

if __name__ == "__main__":

    result_file = (
        "results/test/scene_000_predictions.npz"
    )

    visualize_saved_result(result_file)
