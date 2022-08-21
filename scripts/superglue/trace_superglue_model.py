import os

import torch

from superglue_wrapper import SuperGlueWrapper as SuperGlue


def main():
    model = SuperGlue({})
    batch_size = 1
    height = 480
    width = 640
    num_keypoints = 382
    data = {}
    for i in range(2):
        data[f"image{i}_shape"] = torch.tensor(
            [batch_size, 1, height, width], dtype=torch.float32
        )
        data[f"scores{i}"] = torch.randn(batch_size, num_keypoints)
        data[f"keypoints{i}"] = torch.randn(batch_size, num_keypoints, 2)
        data[f"descriptors{i}"] = torch.randn(batch_size, 256, num_keypoints)

    traced_script_module = torch.jit.trace(model, data)
    traced_script_module.save("superglue_model.pt")
    print(f"\nsuperglue model is saved to: {os.getcwd()}/superglue_model.pt")


if __name__ == "__main__":
    main()
