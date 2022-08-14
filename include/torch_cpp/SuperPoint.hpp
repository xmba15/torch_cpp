/**
 * @file    SuperPoint.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <string>

#include <opencv2/opencv.hpp>

namespace _cv
{
class CV_EXPORTS_W SuperPoint : public cv::Feature2D
{
 public:
    static constexpr int IMAGE_HEIGHT = 480;
    static constexpr int IMAGE_WIDTH = 640;

    struct Param {
        std::string pathToWeights = "";
        int borderRemove = 4;
        float confidenceThresh = 0.015;
        bool alignCorners = true;
        int distThresh = 2;  // nms. set value <= 0 to deactivate nms
        int gpuIdx = -1;     // use gpu >= 0 to specify cuda device
    };

    CV_WRAP static cv::Ptr<SuperPoint> create(const Param& param);
};
}  // namespace _cv
// torch::Device(torch::kCUDA, 0)
