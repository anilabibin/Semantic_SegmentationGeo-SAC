
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyHeads(nn.Module):

    def __init__(
        self,
        hidden_dim=128,
        num_primitives=4
    ):
        super().__init__()

        # Occupancy prediction
        self.occupancy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Primitive prediction
        self.primitive_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_primitives)
        )

        # Boundary prediction
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Cross-view feature projection
        self.view_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(
        self,
        h,
        edge_index
    ):

        src, dst = edge_index

        # Occupancy logits
        occupancy = self.occupancy_head(h)

        # Primitive logits
        primitive = self.primitive_head(h)

        # Boundary prediction for graph edges
        edge_features = torch.cat(
            [h[src], h[dst]], dim=-1
        )

        boundary = self.boundary_head(
            edge_features
        )

        # Projected node representations
        view_features = self.view_projection(h)

        return {
            "occupancy": occupancy,
            "primitive": primitive,
            "boundary": boundary,
            "view_features": view_features
        }


def cross_view_consistency(
    features_1,
    features_2,
    temperature=1.0
):
    """
    Consistency between corresponding superpoints
    from two augmented views.

    Both feature matrices must follow the same
    correspondence ordering.
    """

    z1 = F.normalize(features_1, dim=-1)
    z2 = F.normalize(features_2, dim=-1)

    return (
        1.0 -
        F.cosine_similarity(
            z1, z2, dim=-1
        )
    ).mean() / temperature
