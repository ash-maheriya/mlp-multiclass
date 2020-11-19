//
// Created by ash on 11/16/20.
//
#pragma once

#include <vector>
#include "neuron.h"
namespace neural_net {
/**
 * The layers that comprise the network
 */
class Layer {
 public:
  Layer(std::vector<std::vector<double>>* weights, bool is_input, bool is_output);

  /**
   * Runs the forward pass of every neuron in the layer and passes the values
   * onto the next layer in the network
   * @param next_layer
   */
  void ForwardPassHidden(Layer* prev_layer, Layer* next_layer);

  void ForwardPassInput(Layer* next_layer);

  void ForwardPassOutput(Layer* prev_layer);

  /**
   * Returns how many neurons are in the layer
   * @return how many neurons are in the layer
   */
  double GetSize() const;

  void UpdateValues();

 private:
  const double kBias = 1;

  std::vector<Neuron> neurons_;
  std::vector<std::vector<double>>* weights_;
  std::vector<double> values_;

  bool is_input_layer_ = false;
  bool is_output_layer_ = false;
};
}