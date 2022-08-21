/**
 * @file    App.cpp
 *
 * @author  btran
 *
 */

#include <torch_cpp/torch_cpp.hpp>

int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cerr << "Usage: [app] [path/to/superglue/weights] [path/to/image1] [path/to/image2]" << std::endl;
        return EXIT_FAILURE;
    }
    const std::string WEIGHTS_PATH = argv[1];
    const std::vector<std::string> IMAGE_PATHS = {argv[2], argv[3]};

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

    _cv::SuperGlue::Param superGlueParam;
    superGlueParam.pathToWeights = WEIGHTS_PATH;
    cv::Ptr<_cv::SuperGlue> superPoint = _cv::SuperGlue::create(superGlueParam);

    return EXIT_SUCCESS;
}
