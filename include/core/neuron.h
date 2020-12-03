//
// Created by ash on 11/15/20.
//

#pragma once

#include <vector>
namespace neural_net {
/**
 * The neurons that comprise the layers in a network
 */
class Neuron {
 public:
  /**
   * Takes in all the inputs and weights from the previous layers and outputs
   * an activation value
   * @param weights the weights from the previous layer
   * @param values the activation values from the previous layer
   * @return the calculated activation value of the neuron
   */
  float ForwardPass(const std::vector<float>& weights, const std::vector<float>& values);

  float GetActivation();

  void CalculateError(float prev_error);

  float GetError();

  void SetError(float error);

  void IncrementDelta(float next_error);

  void CalculateGradient(size_t batch_size);

  float GetGradient();

  void SetActivation(float activation);

 private:
  /**
   * Sigmoid activation function maps the given value on the sigmoid graph
   * @param value value to be mapped
   * @return the adjusted value
   */
  float Sigmoid(float value);
  float activation_ = 0;
  float error_;
  float delta_;
  float gradient_;
};
} // namespace neural_net
