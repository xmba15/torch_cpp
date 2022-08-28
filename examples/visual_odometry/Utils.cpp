/**
 * @file    Utils.cpp
 *
 * @author  btran
 *
 */

#include "Utils.hpp"

namespace _cv
{
std::vector<std::string> splitByDelim(const std::string& s, const char delimiter)
{
    std::stringstream ss(s);
    std::string token;
    std::vector<std::string> tokens;
    while (std::getline(ss, token, delimiter)) {
        tokens.emplace_back(token);
    }
    return tokens;
}

std::vector<std::string> readLines(const std::string& textFile)
{
    std::ifstream inFile;
    inFile.open(textFile);

    if (!inFile) {
        throw std::runtime_error("unable to open " + textFile + "\n");
    }

    std::stringstream buffer;
    buffer << inFile.rdbuf();

    return splitByDelim(buffer.str(), '\n');
}

std::vector<cv::Affine3d> parseKittiOdometryPosesGT(const std::string& pathToGT)
{
    auto lines = readLines(pathToGT);
    if (lines.empty()) {
        throw std::runtime_error("file is empty: " + pathToGT);
    }

    std::vector<cv::Affine3d> result;
    for (auto& line : lines) {
        auto strValues = splitByDelim(line, ' ');
        std::vector<double> values;
        std::transform(strValues.begin(), strValues.end(), std::back_inserter(values),
                       [](const auto& strValue) { return std::atof(strValue.c_str()); });
        cv::Mat rotationMat(3, 3, CV_64FC1);
        cv::Mat translationVec(3, 1, CV_64FC1);
        std::copy(values.begin(), values.begin() + 9, rotationMat.ptr<double>());
        std::copy(values.begin() + 9, values.end(), translationVec.ptr<double>());
        result.emplace_back(rotationMat, translationVec);
    }
    return result;
}
}  // namespace _cv
