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
  double ForwardPass(const std::vector<double>& weights, const std::vector<double>& values);

  double GetActivation();

  void CalculateError(double prev_error);

  double GetError();

  void SetError(double error);

  void IncrementDelta(double next_error);

  void CalculateGradient(size_t batch_size);

  void GetGradient();

 private:
  /**
   * Sigmoid activation function maps the given value on the sigmoid graph
   * @param value value to be mapped
   * @return the adjusted value
   */
  double Sigmoid(double value);
  double activation_ = 0;
  double error_;
  double delta_;
  double gradient_;
};
} // namespace neural_net
