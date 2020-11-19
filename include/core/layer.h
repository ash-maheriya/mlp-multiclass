//
// Created by ash on 11/16/20.
//
#pragma once

#include <vector>
namespace neural_net {
/**
 * The layers that comprise the network
 */
class Layer {
 public:
  Layer(std::vector<std::vector<double>>* weights);

  /**
   * Runs the forward pass of every neuron in the layer and passes the values
   * onto the next layer in the network
   * @param next_layer
   */
  void ForwardPass(Layer* next_layer);

  /**
   * Returns how many neurons are in the layer
   * @return how many neurons are in the layer
   */
  double GetSize() const;

 private:
  std::vector<std::vector<double>>* weights_;
  std::vector<double> values_;
  const double kBias = 1;
};
}