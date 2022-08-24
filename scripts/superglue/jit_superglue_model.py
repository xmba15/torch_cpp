import os

import torch

from SuperGluePretrainedNetwork.models.superglue import SuperGlue


def main():
    superglue = SuperGlue({"weights": "outdoor"}).eval()

    scripted_module = torch.jit.script(superglue)
    scripted_module.save("superglue_model.pt")
    print(f"\nsuperglue model is saved to: {os.getcwd()}/superglue_model.pt")


if __name__ == "__main__":
    main()
