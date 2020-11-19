//
// Created by ash on 11/15/20.
//

#pragma once

#include <vector>
namespace neural_net {
class Neuron {
 public:
  // forward propagation (takes input and output states as arguments and computes output state from the input state)
  double ForwardPass(const std::vector<double>& weights, const std::vector<double>& values);

 private:
  double Sigmoid(double value);
  double activation_ = 0;
};
} // namespace neural_net
