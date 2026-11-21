
import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralConsistencyLoss(nn.Module):

    def __init__(
        self,
        lambda_occ=0.5,
        lambda_prim=0.5,
        lambda_boundary=0.5,
        lambda_consistency=0.2,
        lambda_rule=0.1
    ):
        super().__init__()

        self.lambda_occ = lambda_occ
        self.lambda_prim = lambda_prim
        self.lambda_boundary = lambda_boundary
        self.lambda_consistency = lambda_consistency
        self.lambda_rule = lambda_rule

    def forward(
        self,
        predictions,
        targets,
        view_features_1=None,
        view_features_2=None,
        rule_loss=None
    ):

        # 1. Semantic classification loss
        semantic_loss = F.cross_entropy(
            predictions["semantic"],
            targets["semantic"].long()
        )

        # 2. Occupancy prediction loss
        occupancy_loss = F.binary_cross_entropy_with_logits(
            predictions["occupancy"].squeeze(-1),
            targets["occupancy"].float()
        )

        # 3. Primitive prediction loss
        primitive_loss = F.cross_entropy(
            predictions["primitive"],
            targets["primitive"].long()
        )

        # 4. Boundary prediction loss
        boundary_loss = F.binary_cross_entropy_with_logits(
            predictions["boundary"].squeeze(-1),
            targets["boundary"].float()
        )

        # 5. Cross-view consistency loss
        if (
            view_features_1 is not None
            and view_features_2 is not None
        ):

            z1 = F.normalize(view_features_1, dim=-1)
            z2 = F.normalize(view_features_2, dim=-1)

            consistency_loss = (
                1 - F.cosine_similarity(
                    z1, z2, dim=-1
                )
            ).mean()

        else:
            consistency_loss = semantic_loss.new_zeros(())

        # 6. Structural rule loss
        if rule_loss is None:
            rule_loss = semantic_loss.new_zeros(())

        # Overall objective
        total_loss = (
            semantic_loss
            + self.lambda_occ * occupancy_loss
            + self.lambda_prim * primitive_loss
            + self.lambda_boundary * boundary_loss
            + self.lambda_consistency * consistency_loss
            + self.lambda_rule * rule_loss
        )

        loss_dictionary = {
            "total": total_loss,
            "semantic": semantic_loss,
            "occupancy": occupancy_loss,
            "primitive": primitive_loss,
            "boundary": boundary_loss,
            "consistency": consistency_loss,
            "rule": rule_loss
        }

        return loss_dictionary
