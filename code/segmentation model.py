
import torch
import torch.nn as nn

from models.primitive_graph_transformer import (
    AdaptivePrimitiveGraphTransformer
)

from models.consistency_heads import (
    ConsistencyHeads
)


class AdaptiveGraphSegmentationModel(nn.Module):

    def __init__(
        self,
        input_dim,
        num_classes=5,
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        num_primitives=4,
        dropout=0.1
    ):
        super().__init__()

        # Adaptive Primitive Graph Transformer
        self.transformer = AdaptivePrimitiveGraphTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

        # Semantic segmentation classifier
        self.semantic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        # Structural consistency heads
        self.consistency_heads = ConsistencyHeads(
            hidden_dim=hidden_dim,
            num_primitives=num_primitives
        )

    def forward(
        self,
        features,
        edge_index,
        edge_features=None
    ):

        # Step 1: Graph transformer
        h = self.transformer(
            features,
            edge_index,
            edge_features
        )

        # Step 2: Semantic prediction
        semantic = self.semantic_head(h)

        # Step 3: Auxiliary predictions
        auxiliary = self.consistency_heads(
            h,
            edge_index
        )

        # Step 4: Combine predictions
        output = {
            "semantic": semantic,
            "occupancy": auxiliary["occupancy"],
            "primitive": auxiliary["primitive"],
            "boundary": auxiliary["boundary"],
            "view_features": auxiliary["view_features"],
            "embeddings": h
        }

        return output
