
import os
import glob
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"D:\Downloads\Stanford3dDataset_v1.2\Stanford3dDataset_v1.2_Version1"

OUTPUT_PATH = r"D:\Semantics_ptcloud_results\processed_data"

# S3DIS semantic classes
CLASS_NAMES = [
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "table",
    "chair",
    "sofa",
    "bookcase",
    "board",
    "clutter"
]

CLASS_MAP = {
    name: i for i, name in enumerate(CLASS_NAMES)
}

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# 1. DATASET LOADER
# ============================================================

def find_rooms(dataset_path):

    rooms = glob.glob(
        os.path.join(
            dataset_path,
            "Area_*",
            "*",
            "Annotations"
        )
    )

    if not rooms:
        raise FileNotFoundError(
            "No S3DIS annotation folders found. "
            "Check the dataset directory."
        )

    return sorted(rooms)


def load_annotation_file(file_path):

    try:
        data = np.loadtxt(file_path, dtype=np.float32)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] < 6:
            print("Skipping invalid file:", file_path)
            return None

        # XYZ coordinates
        xyz = data[:, :3]

        # RGB values
        rgb = data[:, 3:6]

        return xyz, rgb

    except Exception as e:

        print("Error reading:", file_path, e)
        return None


def load_room(annotation_folder):

    all_points = []
    all_colors = []
    all_labels = []

    annotation_files = glob.glob(
        os.path.join(annotation_folder, "*.txt")
    )

    for file_path in annotation_files:

        filename = Path(file_path).stem

        # Example: chair_1.txt -> chair
        class_name = filename.rsplit("_", 1)[0].lower()

        # Unknown objects are assigned to clutter
        label = CLASS_MAP.get(
            class_name,
            CLASS_MAP["clutter"]
        )

        result = load_annotation_file(file_path)

        if result is None:
            continue

        xyz, rgb = result

        all_points.append(xyz)
        all_colors.append(rgb)

        all_labels.append(
            np.full(len(xyz), label, dtype=np.int64)
        )

    if not all_points:
        return None

    points = np.vstack(all_points)
    colors = np.vstack(all_colors)
    labels = np.concatenate(all_labels)

    return points, colors, labels


# ============================================================
# 2. DATA PREPROCESSING
# ============================================================

def clean_point_cloud(points, colors, labels):

    # Remove NaN and infinite values
    valid = (
        np.isfinite(points).all(axis=1) &
        np.isfinite(colors).all(axis=1)
    )

    points = points[valid]
    colors = colors[valid]
    labels = labels[valid]

    return points, colors, labels


def normalize_data(points, colors):

    # Center coordinates
    center = points.mean(axis=0)

    normalized_points = points - center

    # Normalize RGB to [0, 1]
    normalized_colors = colors / 255.0

    return normalized_points, normalized_colors


def voxel_downsample(points, colors, labels, voxel_size=0.02):

    if voxel_size <= 0:
        return points, colors, labels

    # Assign each point to a voxel
    voxel_indices = np.floor(
        points / voxel_size
    ).astype(np.int64)

    _, inverse = np.unique(
        voxel_indices,
        axis=0,
        return_inverse=True
    )

    number_of_voxels = inverse.max() + 1

    # Average XYZ and RGB within each voxel
    counts = np.bincount(inverse)

    new_points = np.zeros(
        (number_of_voxels, 3), dtype=np.float32
    )

    new_colors = np.zeros(
        (number_of_voxels, 3), dtype=np.float32
    )

    for dimension in range(3):

        new_points[:, dimension] = (
            np.bincount(
                inverse,
                weights=points[:, dimension]
            ) / counts
        )

        new_colors[:, dimension] = (
            np.bincount(
                inverse,
                weights=colors[:, dimension]
            ) / counts
        )

    # Majority semantic label within each voxel
    new_labels = np.zeros(
        number_of_voxels, dtype=np.int64
    )

    for voxel_id in range(number_of_voxels):

        voxel_labels = labels[inverse == voxel_id]

        new_labels[voxel_id] = Counter(
            voxel_labels
        ).most_common(1)[0][0]

    return new_points, new_colors, new_labels


def preprocess_room(points, colors, labels, voxel_size=0.02):

    points, colors, labels = clean_point_cloud(
        points, colors, labels
    )

    points, colors = normalize_data(
        points, colors
    )

    points, colors, labels = voxel_downsample(
        points, colors, labels, voxel_size
    )

    return points, colors, labels


# ============================================================
# 3. DATA AUGMENTATION
# ============================================================

def augment_point_cloud(
    points,
    colors,
    labels,
    rotation=True,
    scaling=True,
    flipping=True,
    jittering=True,
    color_jitter=True
):

    points = points.copy()
    colors = colors.copy()
    labels = labels.copy()

    # Random rotation around Z axis
    if rotation:

        angle = np.random.uniform(
            0, 2 * np.pi
        )

        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ])

        points = points @ rotation_matrix.T

    # Random scaling
    if scaling:

        scale = np.random.uniform(
            0.9, 1.1
        )

        points *= scale

    # Random horizontal flipping
    if flipping:

        if np.random.rand() > 0.5:
            points[:, 0] *= -1

    # Gaussian coordinate jitter
    if jittering:

        noise = np.random.normal(
            0, 0.005, points.shape
        )

        points += noise.astype(np.float32)

    # RGB colour augmentation
    if color_jitter:

        color_noise = np.random.normal(
            0, 0.02, colors.shape
        )

        colors = np.clip(
            colors + color_noise,
            0, 1
        )

    return points, colors, labels


# ============================================================
# 4. PROCESS COMPLETE DATASET
# ============================================================

def process_dataset(
    dataset_path,
    output_path,
    voxel_size=0.02
):

    os.makedirs(output_path, exist_ok=True)

    room_folders = find_rooms(dataset_path)

    print("\nTotal rooms found:", len(room_folders))

    for room_number, annotation_folder in enumerate(
        room_folders, 1
    ):

        room_name = os.path.basename(
            os.path.dirname(annotation_folder)
        )

        area_name = os.path.basename(
            os.path.dirname(
                os.path.dirname(annotation_folder)
            )
        )

        print("\n--------------------------------")
        print("Processing:", area_name, room_name)
        print("--------------------------------")

        result = load_room(annotation_folder)

        if result is None:
            print("No valid points. Skipping room.")
            continue

        points, colors, labels = result

        print("Original points:", len(points))

        points, colors, labels = preprocess_room(
            points,
            colors,
            labels,
            voxel_size
        )

        print("Processed points:", len(points))

        print("Class distribution:")

        for class_id, class_name in enumerate(CLASS_NAMES):

            count = np.sum(labels == class_id)

            print(
                f"{class_name}: {count}"
            )

        # Save processed room
        output_file = os.path.join(
            output_path,
            f"{area_name}_{room_name}.npz"
        )

        np.savez_compressed(
            output_file,
            points=points.astype(np.float32),
            colors=colors.astype(np.float32),
            labels=labels.astype(np.int64)
        )

        print("Saved:", output_file)

    print("\n================================")
    print("DATASET PREPROCESSING COMPLETED")
    print("Output directory:", output_path)
    print("================================")


# ============================================================
# 5. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.02
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    process_dataset(
        DATASET_PATH,
        OUTPUT_PATH,
        voxel_size=args.voxel_size
    )
