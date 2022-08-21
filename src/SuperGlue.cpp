/**
 * @file    SuperGlue.cpp
 *
 * @author  btran
 *
 */

#include <torch/torch.h>

#include <torch_cpp/SuperGlue.hpp>
#include <torch_cpp/Utility.hpp>

namespace _cv
{
class SuperGlueImpl : public SuperGlue
{
 public:
    SuperGlueImpl(const SuperGlue::Param& param);

    // void match(cv::InputArray queryDescriptors, cv::InputArray trainDescriptors,
    //            CV_OUT std::vector<cv::DMatch>& matches, cv::InputArray masks = cv::noArray()) const;

 private:
    SuperGlue::Param m_param;
    torch::Device m_device;
};

cv::Ptr<SuperGlue> SuperGlue::create(const Param& param)
{
    return cv::makePtr<SuperGlueImpl>(param);
}

SuperGlueImpl::SuperGlueImpl(const SuperGlue::Param& param)
    : m_param(param)
    , m_device(torch::kCPU)
{
    if (m_param.pathToWeights.empty()) {
        throw std::runtime_error("empty path to weights");
    }
#if ENABLE_GPU
    if (!torch::cuda::is_available() && m_param.gpuIdx >= 0) {
        DEBUG_LOG("torch does not recognize cuda device so fall back to cpu...");
        m_param.gpuIdx = -1;
    }
#else
    DEBUG_LOG("gpu option is not enabled...");
    m_param.gpuIdx = -1;
#endif

    if (m_param.gpuIdx >= 0) {
        m_device = torch::Device(torch::kCUDA, m_param.gpuIdx);
    }
}
}  // namespace _cv
