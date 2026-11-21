
import os
import glob
import argparse

import numpy as np
import torch

from models.segmentation_model import (
    AdaptiveGraphSegmentationModel
)

from results.metrics import (
    calculate_metrics,
    print_metrics,
    save_metrics
)


# --------------------------------------------------
# 1. LOAD TEST DATA
# --------------------------------------------------

def load_scenes(folder):

    files = sorted(
        glob.glob(os.path.join(folder, "*.pt"))
    )

    if not files:
        raise FileNotFoundError(
            f"No test scenes found in {folder}"
        )

    scenes = []

    for file in files:

        data = torch.load(
            file,
            map_location="cpu",
            weights_only=False
        )

        scenes.append((file, data))

    return scenes


# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------

def load_model(checkpoint_path, device):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model = AdaptiveGraphSegmentationModel(
        input_dim=checkpoint["input_dim"],
        num_classes=checkpoint["num_classes"],
        hidden_dim=checkpoint["hidden_dim"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"]
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


# --------------------------------------------------
# 3. EVALUATION
# --------------------------------------------------

@torch.no_grad()
def evaluate(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Evaluation device:", device)

    os.makedirs(args.output_dir, exist_ok=True)

    model, checkpoint = load_model(
        args.checkpoint,
        device
    )

    scenes = load_scenes(args.test_dir)

    all_predictions = []
    all_targets = []

    for scene_id, (filename, data) in enumerate(scenes):

        features = data["features"].float().to(device)

        edge_index = data["edge_index"].long().to(device)

        edge_features = data["edge_features"].float().to(device)

        targets = data["semantic"].long().numpy()

        outputs = model(
            features,
            edge_index,
            edge_features
        )

        superpoint_predictions = (
            outputs["semantic"]
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

        all_predictions.append(superpoint_predictions)
        all_targets.append(targets)

        # Map superpoint labels to original points
        if (
            "points" in data
            and "point_to_superpoint" in data
        ):

            points = data["points"].numpy()

            point_mapping = (
                data["point_to_superpoint"].numpy()
            )

            point_predictions = (
                superpoint_predictions[point_mapping]
            )

            point_targets = (
                targets[point_mapping]
            )

            np.savez_compressed(
                os.path.join(
                    args.output_dir,
                    f"scene_{scene_id:03d}_predictions.npz"
                ),
                points=points,
                predictions=point_predictions,
                targets=point_targets
            )

        print(
            f"Processed scene: {os.path.basename(filename)}"
        )

    # Overall superpoint-level metrics
    y_pred = np.concatenate(all_predictions)
    y_true = np.concatenate(all_targets)

    metrics = calculate_metrics(
        y_true,
        y_pred,
        checkpoint["num_classes"]
    )

    print("\nTEST RESULTS")
    print_metrics(metrics)

    save_metrics(
        metrics,
        os.path.join(args.output_dir, "test_metrics.json")
    )

    print("\nEvaluation completed.")


# --------------------------------------------------
# 4. MAIN
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_dir",
        type=str,
        default="data/test"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/test"
    )

    args = parser.parse_args()

    evaluate(args)
