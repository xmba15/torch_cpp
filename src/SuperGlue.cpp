/**
 * @file    SuperGlue.cpp
 *
 * @author  btran
 *
 */

#include <torch/script.h>
#include <torch/torch.h>

#include <torch_cpp/SuperGlue.hpp>
#include <torch_cpp/Utility.hpp>

namespace _cv
{
class SuperGlueImpl : public SuperGlue
{
 public:
    explicit SuperGlueImpl(const SuperGlue::Param& param);

    void match(cv::InputArray _queryDescriptors, const std::vector<cv::KeyPoint>& queryKeypoints,
               const cv::Size& querySize, cv::InputArray _trainDescriptors,
               const std::vector<cv::KeyPoint>& trainKeypoints, const cv::Size& trainSize,
               CV_OUT std::vector<cv::DMatch>& matches) const final;

 private:
    SuperGlue::Param m_param;
    torch::Device m_device;
    mutable torch::jit::script::Module m_module;
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
    m_module = torch::jit::load(m_param.pathToWeights);

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
    m_module.eval();
    m_module.to(m_device);
}

void SuperGlueImpl::match(cv::InputArray _queryDescriptors, const std::vector<cv::KeyPoint>& queryKeypoints,
                          const cv::Size& querySize, cv::InputArray _trainDescriptors,
                          const std::vector<cv::KeyPoint>& trainKeypoints, const cv::Size& trainSize,
                          CV_OUT std::vector<cv::DMatch>& matches) const
{
    torch::Dict<std::string, torch::Tensor> data;
    data.insert(
        "image0_shape",
        torch::from_blob(
            std::vector<float>{1, 1, static_cast<float>(querySize.height), static_cast<float>(querySize.width)}.data(),
            {4}, torch::kFloat));
    data.insert(
        "image1_shape",
        torch::from_blob(
            std::vector<float>{1, 1, static_cast<float>(trainSize.height), static_cast<float>(trainSize.width)}.data(),
            {4}, torch::kFloat));

    int numQueryKeyPoints = queryKeypoints.size();
    int numTrainKeyPoints = trainKeypoints.size();

    torch::Tensor descriptors0 =
        torch::from_blob(_queryDescriptors.getMat().ptr<float>(),
                         {1, numQueryKeyPoints, _queryDescriptors.getMat().cols}, torch::kFloat);
    torch::Tensor descriptors1 =
        torch::from_blob(_trainDescriptors.getMat().ptr<float>(),
                         {1, numTrainKeyPoints, _trainDescriptors.getMat().cols}, torch::kFloat);
    descriptors0 = descriptors0.permute({0, 2, 1}).contiguous();
    descriptors1 = descriptors1.permute({0, 2, 1}).contiguous();
    data.insert("descriptors0", std::move(descriptors0));
    data.insert("descriptors1", std::move(descriptors1));

    auto keyPoints0 = torch::zeros({1, numQueryKeyPoints, 2});
    auto scores0 = torch::zeros({1, numQueryKeyPoints});
    for (int i = 0; i < numQueryKeyPoints; ++i) {
        keyPoints0[0][i][0] = queryKeypoints[i].pt.y;
        keyPoints0[0][i][1] = queryKeypoints[i].pt.x;
        scores0[0][i] = queryKeypoints[i].response;
    }
    auto keyPoints1 = torch::zeros({1, numTrainKeyPoints, 2});
    auto scores1 = torch::zeros({1, numTrainKeyPoints});
    for (int i = 0; i < numTrainKeyPoints; ++i) {
        keyPoints1[0][i][0] = trainKeypoints[i].pt.y;
        keyPoints1[0][i][1] = trainKeypoints[i].pt.x;
        scores1[0][i] = trainKeypoints[i].response;
    }
    data.insert("keypoints0", std::move(keyPoints0));
    data.insert("scores0", std::move(scores0));
    data.insert("keypoints1", std::move(keyPoints1));
    data.insert("scores1", std::move(scores1));

    if (!m_device.is_cpu()) {
        for (auto it = data.begin(); it != data.end(); ++it) {
            data.insert(it->key(), it->value().to(m_device));
        }
    }

    auto outputs = c10::impl::toTypedDict<std::string, torch::Tensor>(m_module.forward({data}).toGenericDict());
    auto matches0 = outputs.at("matches0");
    auto matches1 = outputs.at("matches1");

    for (int i = 0; i < numQueryKeyPoints; ++i) {
        if (matches0[0][i].item<std::int64_t>() < 0) {
            continue;
        }
        cv::DMatch match;
        match.imgIdx = 0;
        match.queryIdx = i;
        match.trainIdx = matches0[0][i].item<std::int64_t>();
        matches.emplace_back(match);
    }
}
}  // namespace _cv
