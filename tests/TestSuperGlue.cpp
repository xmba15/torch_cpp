/**
 * @file    TestSuperGlue.cpp
 *
 * @author  btran
 *
 */

#include <gtest/gtest.h>

#include <torch_cpp/torch_cpp.hpp>

#include "config.h"

TEST(TestSuperGlue, TestInitializationFailure)
{
    _cv::SuperGlue::Param param;
    EXPECT_ANY_THROW({ cv::Ptr<_cv::SuperGlue> superGlue = _cv::SuperGlue::create(param); });
}

TEST(TestSuperGlue, TestInitializationSuccess)
{
    _cv::SuperGlue::Param param;
    param.pathToWeights = std::string(DATA_PATH) + "/superglue_model.pt";
    EXPECT_NO_THROW({ cv::Ptr<_cv::SuperGlue> superGlue = _cv::SuperGlue::create(param); });
}
