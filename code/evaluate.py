
import os
import glob
import argparse
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from models.segmentation_model import (
    AdaptiveGraphSegmentationModel
)

from models.losses import (
    StructuralConsistencyLoss
)

from results.metrics import calculate_metrics


# --------------------------------------------------
# 1. REPRODUCIBILITY
# --------------------------------------------------

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# 2. DATASET
# --------------------------------------------------

class GraphDataset(Dataset):

    def __init__(self, folder):

        self.files = sorted(
            glob.glob(os.path.join(folder, "*.pt"))
        )

        if not self.files:
            raise FileNotFoundError(
                f"No .pt files found in {folder}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        data = torch.load(
            self.files[index],
            map_location="cpu",
            weights_only=False
        )

        required = [
            "features",
            "edge_index",
            "edge_features",
            "semantic",
            "occupancy",
            "primitive",
            "boundary"
        ]

        for key in required:
            if key not in data:
                raise KeyError(
                    f"{key} missing in {self.files[index]}"
                )

        return data


# --------------------------------------------------
# 3. DEVICE
# --------------------------------------------------

def get_device():

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# --------------------------------------------------
# 4. FEATURE AUGMENTATION
# --------------------------------------------------

def augment_features(features):

    # Small feature perturbation
    noise = torch.randn_like(features) * 0.01

    augmented = features + noise

    # Random feature dropout
    mask = (
        torch.rand_like(augmented) > 0.05
    ).float()

    return augmented * mask


# --------------------------------------------------
# 5. VALIDATION
# --------------------------------------------------

@torch.no_grad()
def validate(model, loader, device, num_classes):

    model.eval()

    all_predictions = []
    all_targets = []

    for data in loader:

        features = data["features"].float().to(device)
        edge_index = data["edge_index"].long().to(device)
        edge_features = data["edge_features"].float().to(device)

        targets = data["semantic"].long().to(device)

        outputs = model(
            features,
            edge_index,
            edge_features
        )

        predictions = outputs["semantic"].argmax(dim=1)

        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    return calculate_metrics(
        targets,
        predictions,
        num_classes
    )


# --------------------------------------------------
# 6. TRAINING FUNCTION
# --------------------------------------------------

def train(args):

    set_seed(args.seed)

    device = get_device()

    print("Using device:", device)

    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset = GraphDataset(args.train_dir)
    val_dataset = GraphDataset(args.val_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False
    )

    # Batch size 1 is used because graph scenes
    # may contain different numbers of nodes and edges.

    sample = train_dataset[0]

    input_dim = sample["features"].shape[1]

    model = AdaptiveGraphSegmentationModel(
        input_dim=input_dim,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    ).to(device)

    criterion = StructuralConsistencyLoss(
        lambda_occ=args.lambda_occ,
        lambda_prim=args.lambda_prim,
        lambda_boundary=args.lambda_boundary,
        lambda_consistency=args.lambda_consistency,
        lambda_rule=args.lambda_rule
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    best_miou = -1.0

    history = []

    # --------------------------------------------------
    # EPOCH LOOP
    # --------------------------------------------------

    for epoch in range(args.epochs):

        model.train()

        total_epoch_loss = 0.0

        for step, data in enumerate(train_loader):

            features = data["features"].float().to(device)
            edge_index = data["edge_index"].long().to(device)
            edge_features = data["edge_features"].float().to(device)

            targets = {
                "semantic": data["semantic"].long().to(device),
                "occupancy": data["occupancy"].float().to(device),
                "primitive": data["primitive"].long().to(device),
                "boundary": data["boundary"].float().to(device)
            }

            optimizer.zero_grad()

            # First view
            outputs = model(
                features,
                edge_index,
                edge_features
            )

            # Second augmented view
            augmented_features = augment_features(features)

            outputs_2 = model(
                augmented_features,
                edge_index,
                edge_features
            )

            # Combined multi-task loss
            losses = criterion(
                outputs,
                targets,
                outputs["view_features"],
                outputs_2["view_features"]
            )

            loss = losses["total"]

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite loss encountered."
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_epoch_loss += loss.item()

        scheduler.step()

        average_loss = (
            total_epoch_loss / len(train_loader)
        )

        # Validation
        val_metrics = validate(
            model,
            val_loader,
            device,
            args.num_classes
        )

        val_miou = val_metrics["mIoU"]

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Loss: {average_loss:.4f} "
            f"OA: {val_metrics['OA']:.4f} "
            f"mIoU: {val_miou:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "loss": average_loss,
            "OA": val_metrics["OA"],
            "mIoU": val_miou
        })

        # Save best model
        if val_miou > best_miou:

            best_miou = val_miou

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "input_dim": input_dim,
                "num_classes": args.num_classes,
                "hidden_dim": args.hidden_dim,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "best_miou": best_miou,
                "seed": args.seed
            }

            torch.save(
                checkpoint,
                os.path.join(
                    args.output_dir,
                    "best_model.pth"
                )
            )

    np.save(
        os.path.join(args.output_dir, "history.npy"),
        history,
        allow_pickle=True
    )

    print("\nTraining completed.")
    print("Best validation mIoU:", best_miou)


# --------------------------------------------------
# 7. CONFIGURATION
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", type=str,
                        default="data/train")

    parser.add_argument("--val_dir", type=str,
                        default="data/validation")

    parser.add_argument("--output_dir", type=str,
                        default="checkpoints")

    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--num_classes", type=int, default=5)

    parser.add_argument("--hidden_dim", type=int, default=128)

    parser.add_argument("--num_heads", type=int, default=4)

    parser.add_argument("--num_layers", type=int, default=3)

    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--weight_decay", type=float, default=0.0001)

    parser.add_argument("--lambda_occ", type=float, default=0.5)

    parser.add_argument("--lambda_prim", type=float, default=0.5)

    parser.add_argument("--lambda_boundary", type=float, default=0.5)

    parser.add_argument("--lambda_consistency", type=float, default=0.2)

    parser.add_argument("--lambda_rule", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train(args)
