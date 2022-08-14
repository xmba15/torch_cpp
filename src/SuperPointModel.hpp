/**
 * @file    SuperPointModel.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <utility>
#include <vector>

#include <torch/torch.h>

namespace _cv
{
struct SuperPointModel : torch::nn::Module {
 public:
    SuperPointModel();

    std::vector<torch::Tensor> forward(torch::Tensor x);

    torch::nn::Conv2d conv1a;
    torch::nn::Conv2d conv1b;

    torch::nn::Conv2d conv2a;
    torch::nn::Conv2d conv2b;

    torch::nn::Conv2d conv3a;
    torch::nn::Conv2d conv3b;

    torch::nn::Conv2d conv4a;
    torch::nn::Conv2d conv4b;

    torch::nn::Conv2d convPa;
    torch::nn::Conv2d convPb;

    // descriptor
    torch::nn::Conv2d convDa;
    torch::nn::Conv2d convDb;

 private:
    static constexpr int c1 = 64;
    static constexpr int c2 = 64;
    static constexpr int c3 = 128;
    static constexpr int c4 = 128;
    static constexpr int c5 = 256;
    static constexpr int d1 = 256;
};
}  // namespace _cv
