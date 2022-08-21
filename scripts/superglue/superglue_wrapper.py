from typing import Any, Dict

import torch

from SuperGluePretrainedNetwork.models.superglue import (
    SuperGlue,
    arange_like,
    log_optimal_transport,
    normalize_keypoints,
)


class SuperGlueWrapper(SuperGlue):
    def __init__(self, config: Dict[str, Any]):
        SuperGlue.__init__(self, config)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        desc0, desc1 = data["descriptors0"], data["descriptors1"]
        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]

        kpts0 = normalize_keypoints(kpts0, data["image0_shape"])
        kpts1 = normalize_keypoints(kpts1, data["image1_shape"])

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, data["scores0"])
        desc1 = desc1 + self.kenc(kpts1, data["scores1"])

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum("bdn,bdm->bnm", mdesc0, mdesc1)
        scores = scores / self.config["descriptor_dim"] ** 0.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score, iters=self.config["sinkhorn_iterations"]
        )

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config["match_threshold"])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return indices0, indices1, mscores0, mscores1
