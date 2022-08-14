/**
 * @file    App.cpp
 *
 * @author  btran
 *
 */

#include <torch_cpp/torch_cpp.hpp>

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: [app] [path/to/superpoint/weights] [path/to/image]" << std::endl;
        return EXIT_FAILURE;
    }
    const std::string WEIGHTS_PATH = argv[1];
    const std::string IMAGE_PATH = argv[2];

    _cv::SuperPoint::Param param;
    param.pathToWeights = WEIGHTS_PATH;
    param.distThresh = 5;
    param.gpuIdx = 0;
    cv::Ptr<cv::Feature2D> superPoint = _cv::SuperPoint::create(param);

    cv::Mat image = cv::imread(IMAGE_PATH);
    if (image.empty()) {
        std::cerr << "failed to read image: " << IMAGE_PATH << std::endl;
        return EXIT_FAILURE;
    }
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat descriptors;
    std::vector<cv::KeyPoint> keyPoints;
    superPoint->detectAndCompute(gray, cv::Mat(), keyPoints, descriptors);

    std::cout << "number of keypoints: " << keyPoints.size() << std::endl;

    cv::drawKeypoints(image, keyPoints, image);
    cv::imshow("superpoint keypoint", image);
    cv::waitKey();

    return EXIT_SUCCESS;
}
