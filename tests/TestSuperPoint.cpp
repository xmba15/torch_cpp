/**
 * @file    TestSuperPoint.cpp
 *
 * @author  btran
 *
 */

#include <gtest/gtest.h>

#include <torch_cpp/torch_cpp.hpp>

#include "config.h"

TEST(TestSuperPoint, TestInitializationFailure)
{
    _cv::SuperPoint::Param param;
    EXPECT_ANY_THROW({ cv::Ptr<cv::Feature2D> superPoint = _cv::SuperPoint::create(param); });
}

TEST(TestSuperPoint, TestInitializationSuccess)
{
    _cv::SuperPoint::Param param;
    param.pathToWeights = std::string(DATA_PATH) + "/superpoint_model.pt";
    EXPECT_NO_THROW({ cv::Ptr<cv::Feature2D> superPoint = _cv::SuperPoint::create(param); });
}

TEST(TestSuperPoint, TestSuperPointDetection)
{
    _cv::SuperPoint::Param param;
    param.pathToWeights = std::string(DATA_PATH) + "/superpoint_model.pt";
    cv::Ptr<cv::Feature2D> superPoint = _cv::SuperPoint::create(param);

    std::string IMAGE_PATH = std::string(DATA_PATH) + "/images/30.jpg";
    cv::Mat image = cv::imread(IMAGE_PATH, 0);
    cv::Mat mask, descriptors;
    std::vector<cv::KeyPoint> keyPoints;
    superPoint->detectAndCompute(image, mask, keyPoints, descriptors);
    EXPECT_GT(keyPoints.size(), 0);

    if (keyPoints.size() > 0) {
        EXPECT_EQ(keyPoints.size(), descriptors.rows);
    }

    // test normalization
    int numKeyPoinst = descriptors.rows;
    cv::Mat squaredSumMat;
    cv::reduce(descriptors.mul(descriptors), squaredSumMat, 1, cv::REDUCE_SUM);
    for (int i = 0; i < numKeyPoinst; ++i) {
        float squaredSum = squaredSumMat.ptr<float>()[i];
        EXPECT_NEAR(squaredSum, 1, 1e-4);
    }
}
