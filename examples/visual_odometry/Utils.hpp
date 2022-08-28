/**
 * @file    Utils.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <opencv2/opencv.hpp>

namespace _cv
{
std::vector<std::string> splitByDelim(const std::string& s, const char delimiter);

std::vector<std::string> readLines(const std::string& textFile);

std::vector<cv::Affine3d> parseKittiOdometryPosesGT(const std::string& pathToGT);
}  // namespace _cv
