import os

import torch

from superpoint_wrapper import SuperPointWrapper as SuperPoint


def main():
    model = SuperPoint({})
    model.eval()
    batch_size = 1
    channel = 1
    height = 480
    width = 640
    x = torch.rand(batch_size, channel, height, width)

    # semi, desc = model(x)
    # print(semi.shape)
    # print(desc.shape)

    traced_script_module = torch.jit.trace(model, x)
    traced_script_module.save("superpoint_model.pt")
    print(f"\nsuperpoint model is saved to: {os.getcwd()}/superpoint_model.pt")


if __name__ == "__main__":
    main()
