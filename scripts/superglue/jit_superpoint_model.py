import os

import torch

from SuperGluePretrainedNetwork.models.superpoint import SuperPoint


def main():
    superpoint = SuperPoint({}).eval()

    scripted_superpoint = torch.jit.script(superpoint)
    scripted_superpoint.save("superpoint_model.pt")
    print(f"\nsuperpoint model is saved to: {os.getcwd()}/superpoint_model.pt")


if __name__ == "__main__":
    main()
