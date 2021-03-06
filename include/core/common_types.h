//
// Created by ash on 11/16/20.
//
#pragma once
#include <vector>

namespace neural_net {
typedef std::vector<std::vector<std::vector<float>>> Weight_Collection_t;
typedef std::vector<float> Error_Collection_t;
typedef std::vector<std::vector<float>> Delta_Collection_t;
typedef std::vector<std::vector<float>> Gradient_Collection_t;
typedef std::vector<std::vector<float>> Image_t;

} // namespace neural_net
