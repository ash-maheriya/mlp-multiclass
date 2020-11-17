//
// Created by ash on 11/15/20.
//

#pragma once

#include <vector>
class Neuron {
 public:
  // forward propagation (takes input and output states as arguments and computes output state from the input state)
  double ForwardPass(std::vector<double> weights, std::vector<double> values);
 private:
  void UpdateValue(std::vector<double> weights, std::vector<double> values);
  double Sigmoid();

  double value_ = 0;
  double activation_ = 0;
};

