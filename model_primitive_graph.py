
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def edge_softmax(scores, dst, num_nodes):
    """
    Softmax over incoming edges for each destination node.

    scores: [E, H]
    dst:    [E]
    """
    num_edges, num_heads = scores.shape

    index = dst.unsqueeze(-1).expand(-1, num_heads)

    max_values = torch.full(
        (num_nodes, num_heads),
        -torch.inf,
        device=scores.device,
        dtype=scores.dtype
    )

    max_values.scatter_reduce_(
        0, index, scores,
        reduce="amax",
        include_self=True
    )

    exp_scores = torch.exp(
        scores - max_values[dst]
    )

    denominator = torch.zeros_like(max_values)

    denominator.index_add_(0, dst, exp_scores)

    attention = exp_scores / (
        denominator[dst] + 1e-12
    )

    return attention


class PrimitiveGraphTransformer(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_heads=4,
        dropout=0.1
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Input feature projection
        self.input_projection = nn.Linear(
            input_dim, hidden_dim
        )

        # Query, Key and Value
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        # Learnable graph relationship weights
        self.graph_weights = nn.Parameter(
            torch.ones(4)
        )

        # Positional and graph relationship encoding
        self.edge_encoder = nn.Sequential(
            nn.Linear(7, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_heads)
        )

        self.output_projection = nn.Linear(
            hidden_dim, hidden_dim
        )

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(
        self,
        x,
        edge_index,
        edge_features=None
    ):
        """
        x:
            [N, input_dim]

        edge_index:
            [2, E]
            edge_index[0] = source nodes
            edge_index[1] = destination nodes

        edge_features:
            [E, 7]
            Columns:
            geometric similarity,
            topological similarity,
            contextual similarity,
            boundary strength,
            relative dx, dy, dz.
        """

        num_nodes = x.size(0)

        src, dst = edge_index

        h = self.input_projection(x)

        # Q, K, V projections
        Q = self.query(h).view(
            num_nodes, self.num_heads, self.head_dim
        )

        K = self.key(h).view(
            num_nodes, self.num_heads, self.head_dim
        )

        V = self.value(h).view(
            num_nodes, self.num_heads, self.head_dim
        )

        # Edge-based attention
        q_src = Q[src]
        k_dst = K[dst]

        scores = (
            q_src * k_dst
        ).sum(dim=-1) / math.sqrt(self.head_dim)

        if edge_features is not None:

            # Four graph relationship terms:
            # geometric, topological, contextual, boundary

            relation_weights = F.softmax(
                self.graph_weights, dim=0
            )

            graph_relations = edge_features[:, :4]

            graph_bias = (
                graph_relations *
                relation_weights.unsqueeze(0)
            ).sum(dim=-1, keepdim=True)

            # Positional encoding
            positional_bias = self.edge_encoder(
                edge_features
            )

            # Boundary suppression
            boundary = edge_features[:, 3:4]

            scores = (
                scores
                + graph_bias
                + positional_bias
                - boundary
            )

        # Attention normalization
        attention = edge_softmax(
            scores, dst, num_nodes
        )

        attention = self.dropout(attention)

        # Message aggregation
        messages = (
            attention.unsqueeze(-1) * V[src]
        )

        aggregated = torch.zeros(
            num_nodes,
            self.num_heads,
            self.head_dim,
            device=x.device,
            dtype=x.dtype
        )

        aggregated.index_add_(0, dst, messages)

        aggregated = aggregated.reshape(
            num_nodes, self.hidden_dim
        )

        # Residual connection
        h = self.norm1(
            h + self.dropout(
                self.output_projection(aggregated)
            )
        )

        # Feed-forward network
        h = self.norm2(
            h + self.dropout(self.ffn(h))
        )

        return h


class AdaptivePrimitiveGraphTransformer(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        dropout=0.1
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            PrimitiveGraphTransformer(
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                num_heads,
                dropout
            )
            for i in range(num_layers)
        ])

    def forward(self, x, edge_index, edge_features=None):

        for layer in self.layers:
            x = layer(
                x,
                edge_index,
                edge_features
            )

        return x
