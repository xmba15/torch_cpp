/**
 * @file    App.cpp
 *
 * @author  btran
 *
 */

#include <torch_cpp/torch_cpp.hpp>

int main(int argc, char* argv[])
{
    if (argc != 5) {
        std::cerr
            << "Usage: [app] [path/to/superpoint/weights] [path/to/superglue/weights] [path/to/image1] [path/to/image2]"
            << std::endl;
        return EXIT_FAILURE;
    }
    const std::string SUPERPOINT_WEIGHTS_PATH = argv[1];
    const std::string SUPERGLUE_WEIGHTS_PATH = argv[2];
    const std::vector<std::string> IMAGE_PATHS = {argv[3], argv[4]};

    std::vector<cv::Mat> images;
    std::vector<cv::Mat> grays;
    std::transform(IMAGE_PATHS.begin(), IMAGE_PATHS.end(), std::back_inserter(images),
                   [](const auto& imagePath) { return cv::imread(imagePath); });
    for (int i = 0; i < 2; ++i) {
        if (images[i].empty()) {
            throw std::runtime_error("failed to open " + IMAGE_PATHS[i]);
        }
    }
    std::transform(IMAGE_PATHS.begin(), IMAGE_PATHS.end(), std::back_inserter(grays),
                   [](const auto& imagePath) { return cv::imread(imagePath, 0); });

    std::vector<std::vector<cv::KeyPoint>> keyPointsList(2);
    std::vector<cv::Mat> descriptorsList(2);
    _cv::SuperPoint::Param superPointParam;
    superPointParam.pathToWeights = SUPERPOINT_WEIGHTS_PATH;
    superPointParam.distThresh = 2;
    superPointParam.borderRemove = 4;
    superPointParam.confidenceThresh = 0.015;
    superPointParam.gpuIdx = 0;

    cv::Ptr<cv::Feature2D> superPoint = _cv::SuperPoint::create(superPointParam);
    for (int i = 0; i < 2; ++i) {
        superPoint->detectAndCompute(grays[i], cv::Mat(), keyPointsList[i], descriptorsList[i]);
    }

    _cv::SuperGlue::Param superGlueParam;
    superGlueParam.pathToWeights = SUPERGLUE_WEIGHTS_PATH;
    superGlueParam.gpuIdx = 0;
    cv::Ptr<_cv::SuperGlue> superGlue = _cv::SuperGlue::create(superGlueParam);
    std::vector<cv::DMatch> matches;
    superGlue->match(descriptorsList[0], keyPointsList[0], images[0].size(), descriptorsList[1], keyPointsList[1],
                     images[1].size(), matches);

    return EXIT_SUCCESS;
}
