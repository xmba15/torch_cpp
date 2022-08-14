/**
 * @file    SuperPoint.cpp
 *
 * @author  btran
 *
 */

#include <memory>

#include <torch_cpp/SuperPoint.hpp>
#include <torch_cpp/Utility.hpp>

#include "SuperPointModel.hpp"

namespace
{
cv::Mat copyRows(const cv::Mat& src, const std::vector<int>& indices);
}  // namespace

namespace _cv
{
class SuperPointImpl : public SuperPoint
{
 public:
    explicit SuperPointImpl(const SuperPoint::Param& param);

    void detectAndCompute(cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint>& keyPoints,
                          cv::OutputArray _descriptors, bool useProvidedKeypoints) CV_OVERRIDE;

    static std::vector<int> nmsFast(const std::vector<cv::KeyPoint>& keyPoints, int height, int width, int distThresh);

    int descriptorSize() const CV_OVERRIDE
    {
        return 256;
    }

    int descriptorType() const CV_OVERRIDE
    {
        return CV_32F;
    }

 private:
    SuperPoint::Param m_param;
    std::shared_ptr<SuperPointModel> m_model;
    torch::Device m_device;
};

cv::Ptr<SuperPoint> SuperPoint::create(const Param& param)
{
    return cv::makePtr<SuperPointImpl>(param);
}

SuperPointImpl::SuperPointImpl(const SuperPoint::Param& param)
    : m_param(param)
    , m_model(std::make_shared<SuperPointModel>())
    , m_device(torch::kCPU)
{
    if (m_param.pathToWeights.empty()) {
        throw std::runtime_error("empty path to weights");
    }
    try {
        torch::load(m_model, m_param.pathToWeights);
    } catch (const std::exception& e) {
        INFO_LOG("%s", e.what());
        exit(1);
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
    DEBUG_LOG("use device: %s", m_device.str().c_str());
    m_model->eval();
    m_model->to(m_device);
}

void SuperPointImpl::detectAndCompute(cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint>& keyPoints,
                                      cv::OutputArray _descriptors, bool useProvidedKeypoints)
{
    cv::Mat image = _image.getMat();
    cv::Mat mask = _mask.getMat();

    if (image.empty() || image.depth() != CV_8U) {
        CV_Error(cv::Error::StsBadArg, "image is empty or has incorrect depth (!=CV_8U)");
    }

    if (!mask.empty() && mask.type() != CV_8UC1) {
        CV_Error(cv::Error::StsBadArg, "mask has incorrect type (!=CV_8UC1)");
    }

    std::vector<torch::Tensor> tensors;
    {
        cv::Mat buffer;
        cv::resize(image, buffer, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), 0, 0, cv::INTER_CUBIC);
        buffer.convertTo(buffer, CV_32FC1, 1 / 255.);

        auto x = torch::from_blob(buffer.ptr<float>(), {1, 1, IMAGE_HEIGHT, IMAGE_WIDTH}, torch::kFloat);
        x = x.set_requires_grad(false);
        x = x.to(m_device);
        tensors = m_model->forward(x);
    }

    // preparing mask
    if (!mask.empty()) {
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(mask, &minVal, &maxVal, &minLoc, &maxLoc);
        if (static_cast<int>(maxVal) == 255) {
            mask /= 255;
        }
        cv::resize(mask, mask, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), 0, 0, cv::INTER_NEAREST);
    }

    if (m_param.borderRemove > 0) {
        cv::Mat borderMask = cv::Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1);
        borderMask(cv::Range(m_param.borderRemove, IMAGE_HEIGHT - m_param.borderRemove),
                   cv::Range(m_param.borderRemove, IMAGE_WIDTH - m_param.borderRemove))
            .setTo(1);
        if (mask.empty()) {
            mask = borderMask.clone();
        } else {
            cv::bitwise_and(mask, borderMask, mask);
        }
    }

    // get keypoints
    torch::Tensor kpts;
    torch::Tensor scores;
    {
        auto heatmap = std::move(tensors[0]).squeeze();  // H x W
        torch::Tensor kptMask = heatmap >= m_param.confidenceThresh;
        if (!mask.empty()) {
            kptMask = kptMask.bitwise_and(
                torch::from_blob(mask.data, {IMAGE_HEIGHT, IMAGE_WIDTH}, torch::kBool).to(m_device));
        }

        kpts = torch::nonzero(kptMask);  // num_keypoints x 2; (y, x)

        if (kpts.size(0) == 0) {
            return;
        }

        scores = heatmap.masked_select(kptMask);
        if (!m_device.is_cpu()) {
            scores = scores.detach().cpu();
        }
    }

    // get descriptors
    {
        torch::Tensor desc = std::move(tensors[1]);  // 1 x 256 x H/8 x W/8

        // normalize keypoints coordinates to -1,1
        auto grid = torch::zeros({1, 1, kpts.size(0), 2}).to(m_device);  // 1 x 1 x num_keypoints, 2
        grid[0][0].slice(1, 0, 1) = 2.0 * kpts.slice(1, 1, 2).to(torch::kFloat) / IMAGE_WIDTH - 1;   // x
        grid[0][0].slice(1, 1, 2) = 2.0 * kpts.slice(1, 0, 1).to(torch::kFloat) / IMAGE_HEIGHT - 1;  // y

        desc = torch::nn::functional::grid_sample(desc, grid,
                                                  torch::nn::functional::GridSampleFuncOptions().align_corners(
                                                      m_param.alignCorners));  // 1 x 256 x 1 x num_keypoints
        desc = desc.squeeze(0).squeeze(1);                                     // 256 x num_keypoints

        desc = torch::nn::functional::normalize(desc, torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));

        desc = desc.permute({1, 0}).contiguous();

        if (!m_device.is_cpu()) {
            kpts = kpts.detach().cpu();  // num_keypoints x 2; (y, x)
            desc = desc.detach().cpu();  // num_keypoints x 256
        }

        _descriptors.create(cv::Size(256, kpts.size(0)), CV_32F, -1, true);
        std::memcpy(_descriptors.getMat().ptr<float>(), desc.data_ptr<float>(), sizeof(float) * desc.numel());
    }

    keyPoints.resize(kpts.size(0));
    for (int i = 0; i < kpts.size(0); ++i) {
        auto& curKeyPoint = keyPoints[i];
        curKeyPoint.pt.y = kpts[i][0].item<float>();
        curKeyPoint.pt.x = kpts[i][1].item<float>();
        curKeyPoint.response = scores[i].item<float>();
    }

    // apply nms
    if (m_param.distThresh > 0) {
        std::vector<int> keepIndices = nmsFast(keyPoints, IMAGE_HEIGHT, IMAGE_WIDTH, m_param.distThresh);
        std::vector<cv::KeyPoint> keepKeyPoints;
        keepKeyPoints.reserve(keepIndices.size());
        std::transform(keepIndices.begin(), keepIndices.end(), std::back_inserter(keepKeyPoints),
                       [&keyPoints](int idx) { return keyPoints[idx]; });
        keyPoints = std::move(keepKeyPoints);
        cv::Mat buffer = ::copyRows(_descriptors.getMat(), keepIndices);
        _descriptors.createSameSize(buffer, CV_32F);
        std::memcpy(_descriptors.getMat().ptr<float>(), buffer.ptr<float>(), sizeof(float) * buffer.rows * buffer.cols);
    }

    for (auto& keyPoint : keyPoints) {
        keyPoint.pt.x *= static_cast<float>(image.cols) / IMAGE_WIDTH;
        keyPoint.pt.y *= static_cast<float>(image.rows) / IMAGE_HEIGHT;
    }
}

std::vector<int> SuperPointImpl::nmsFast(const std::vector<cv::KeyPoint>& keyPoints, int height, int width,
                                         int distThresh)
{
    static constexpr int TO_PROCESS = 0;
    static constexpr int EMPTY_OR_SUPPRESSED = 1;

    std::vector<int> sortedIndices(keyPoints.size());
    std::iota(sortedIndices.begin(), sortedIndices.end(), 0);

    // sort in descending order base on confidence
    std::stable_sort(sortedIndices.begin(), sortedIndices.end(),
                     [&keyPoints](int lidx, int ridx) { return keyPoints[lidx].response > keyPoints[ridx].response; });

    cv::Mat grid = cv::Mat(height, width, CV_8U, TO_PROCESS);

    std::vector<int> keepIndices;

    for (int idx : sortedIndices) {
        int x = keyPoints[idx].pt.x;
        int y = keyPoints[idx].pt.y;

        if (grid.ptr<uchar>(y)[x] != TO_PROCESS) {
            continue;
        }

        for (int i = y - distThresh; i < y + distThresh; ++i) {
            if (i < 0 || i >= height) {
                continue;
            }

            for (int j = x - distThresh; j < x + distThresh; ++j) {
                if (j < 0 || j >= width) {
                    continue;
                }
                grid.ptr<uchar>(i)[j] = EMPTY_OR_SUPPRESSED;
            }
        }
        keepIndices.emplace_back(idx);
    }

    return keepIndices;
}

}  // namespace _cv

namespace
{
cv::Mat copyRows(const cv::Mat& src, const std::vector<int>& indices)
{
    int newNumRows = indices.size();
    cv::Mat dst(0, newNumRows, src.type());
    for (int i = 0; i < newNumRows; ++i) {
        dst.push_back(src.row(indices[i]));
    }
    return dst;
}
}  // namespace
