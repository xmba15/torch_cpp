/**
 * @file    SuperGlue.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <opencv2/opencv.hpp>

namespace _cv
{
class SuperGlue
{
 public:
    struct Param {
        std::string pathToWeights = "";
        int gpuIdx = -1;  // use gpu >= 0 to specify cuda device
    };

    static cv::Ptr<SuperGlue> create(const Param& param);
};
}  // namespace _cv
