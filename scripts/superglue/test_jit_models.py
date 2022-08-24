import os
from typing import List

import cv2
import numpy as np
import torch

from SuperGluePretrainedNetwork.models.utils import read_image


def get_args():
    import argparse

    parser = argparse.ArgumentParser("")
    parser.add_argument("--superpoint", "-sp", type=str, required=True)
    parser.add_argument("--superglue", "-sg", type=str, required=True)
    parser.add_argument("--image0", "-i0", type=str, required=True)
    parser.add_argument("--image1", "-i1", type=str, required=True)
    parser.add_argument("--match_threshold", "-m", type=float, default=0.2)
    parser.add_argument("--keypoint_threshold", "-k", type=float, default=0.2)
    parser.add_argument("--remove_borders", "-r", type=int, default=4)
    parser.add_argument("--nms_radius", "-n", type=int, default=2)

    return parser.parse_args()


def main(args):
    image_paths = [args.image0, args.image1]

    superpoint = torch.jit.load(args.superpoint)
    superglue = torch.jit.load(args.superglue)

    input_data = [
        read_image(
            image_path, device="cpu", resize=[640, 480], rotation=0, resize_float=False
        )
        for image_path in image_paths
    ]

    images = [input[0] for input in input_data]
    for image in images:
        assert image is not None
    inps = [input[1] for input in input_data]

    batch_inp = torch.cat(inps, dim=0)
    batch_result = superpoint(
        {
            "image": batch_inp,
            "keypoint_threshold": torch.Tensor([args.keypoint_threshold]),
            "remove_borders": torch.LongTensor([args.remove_borders]),
            "nms_radius": torch.LongTensor([args.nms_radius]),
        }
    )

    for (key, val) in batch_result.items():
        print(key, val[0].shape, val[0].dtype)

    kptsList: List[List[cv2.KeyPoint]] = []
    data = {}
    for i in range(2):
        data[f"image{i}_shape"] = torch.tensor([1, 1, *images[i].shape[:2]])
        data[f"descriptors{i}"] = batch_result["descriptors"][i].unsqueeze(0)
        data[f"keypoints{i}"] = batch_result["keypoints"][i].unsqueeze(0)
        data[f"scores{i}"] = batch_result["scores"][i].unsqueeze(0)

        kpts = []
        for point in batch_result["keypoints"][i].tolist():
            new_keypoint = cv2.KeyPoint()
            new_keypoint.pt = (point[0], point[1])
            kpts.append(new_keypoint)
        kptsList.append(kpts)
    data["match_threshold"] = torch.Tensor([args.match_threshold])

    output = superglue(data)

    print("\n")
    for (key, val) in output.items():
        print(key, val.shape, val.dtype)

    output["matches0"] = output["matches0"].squeeze(0).tolist()
    output["matching_scores0"] = output["matching_scores0"].squeeze(0).tolist()

    matches: List[cv2.DMatch] = []
    for i, (match_idx, score) in enumerate(
        zip(output["matches0"], output["matching_scores0"])
    ):
        if match_idx < 0:
            continue
        new_match = cv2.DMatch()
        new_match.imgIdx = 1
        new_match.queryIdx = i
        new_match.trainIdx = match_idx
        matches.append(new_match)

    matched_image = cv2.drawMatches(
        images[0].astype(np.uint8),
        kptsList[0],
        images[1].astype(np.uint8),
        kptsList[1],
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite("matches.jpg", matched_image)


if __name__ == "__main__":
    main(get_args())
