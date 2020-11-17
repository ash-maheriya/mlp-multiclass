//
// Created by ash on 11/15/20.
//

#include "core/neuron.h"

#include <stdio.h>
#include <stdexcept>
#include <vector>

namespace neural_net {

double Neuron::ForwardPass(std::vector<double> weights,
                           std::vector<double> values) {
  UpdateValue(weights, values);
  return Sigmoid();
}

void Neuron::UpdateValue(std::vector<double> weights,
                         std::vector<double> values) {
  if (weights.size() != values.size()) {
    throw std::invalid_argument(
        "Must have the same number of weights and values");
  }
  for (size_t i = 0; i < weights.size(); i++) {
    value_ += weights[i] * values[i];
  }
  if (value_ < 0) {
    throw std::exception();
  }
}

double Neuron::Sigmoid() {
  activation_ = (value_) / (1 + value_);
  return activation_;
}
}A