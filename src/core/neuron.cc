//
// Created by ash on 11/15/20.
//

#include "core/neuron.h"

#include <math.h>
#include <stdio.h>

#include <stdexcept>
#include <vector>

namespace neural_net {

double Neuron::ForwardPass(const std::vector<double>& weights, const std::vector<double>& values) {
  double value = 0;
  if (weights.size() != values.size()) {
    throw std::invalid_argument(
        "Must have the same number of weights and values");
  }
  for (size_t i = 0; i < weights.size(); i++) {
    value += weights[i] * values[i];
  }
  return Sigmoid(value);
}

double Neuron::Sigmoid(double value) {
  activation_ = 1.0 / (1.0 + exp(-1.0*value));
  return activation_;
}
double Neuron::GetActivation() {
  return activation_;
}

void Neuron::CalculateError(double prev_error) {
  error_ = prev_error * activation_ * (1.0 - activation_);
}
double Neuron::GetError() {
  return error_;
}

void Neuron::SetError(double error) {
  error_ = error;
}
void Neuron::IncrementDelta(double next_error) {
  delta_ += activation_*next_error;
}
void Neuron::CalculateGradient(size_t batch_size) {
  gradient_ = delta_/(float)batch_size;
}
}  // namespace neural_net