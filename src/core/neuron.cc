//
// Created by ash on 11/15/20.
//

#include "core/neuron.h"

#include <math.h>
#include <stdio.h>

#include <iostream>
#include <stdexcept>
#include <vector>

namespace neural_net {

float Neuron::ForwardPass(const std::vector<float>& weights, const std::vector<float>& values) {
  float value = 0;
  if (weights.size() != values.size()) {
    throw std::invalid_argument(
        "Must have the same number of weights and values");
  }
  for (size_t i = 0; i < weights.size(); i++) {
    value += weights[i] * values[i];
  }
  activation_ = Sigmoid(value);
  return Sigmoid(value);
}

float Neuron::Sigmoid(float value) {
  return 1.0 / (1.0 + exp(-1.0*value));
}
float Neuron::GetActivation() {
  return activation_;
}

void Neuron::CalculateError(float prev_error) {
  error_ = prev_error * activation_ * (1.0 - activation_);
}
float Neuron::GetError() {
  return error_;
}

void Neuron::SetError(float error) {
  error_ = error;
}
void Neuron::IncrementDelta(float next_error) {
  delta_ += activation_*next_error;
}
void Neuron::CalculateGradient(size_t batch_size) {
  gradient_ = delta_/(float)batch_size;
}
float Neuron::GetGradient() {
  return gradient_;
}
void Neuron::SetActivation(float activation) {
  activation_ = activation;
}
void Neuron::ResetDelta() {
  delta_ = 0;
}
}  // namespace neural_net