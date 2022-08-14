from typing import Dict

import torch

from SuperGluePretrainedNetwork.models.superpoint import SuperPoint


class SuperPointWrapper(SuperPoint):
    def __init__(self, config: Dict):
        SuperPoint.__init__(self, config)

    def forward(self, x: torch.Tensor):
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)  # B x 65 x H/8 x W/8
        semi = torch.nn.functional.softmax(semi, 1)[:, :-1]
        b, _, h, w = semi.shape
        semi = semi.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        semi = semi.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)

        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)  # B x 256 x H/8 x W/8
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        return semi, desc
