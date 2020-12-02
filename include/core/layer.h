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

  double ForwardPassOutput(Layer* prev_layer);

  /**
   * Returns how many neurons are in the layer
   * @return how many neurons are in the layer
   */
  double GetSize() const;

  void CalculateErrors(std::vector<double> next_errors);

  void CalculateOutputError(size_t label);

  void UpdateValues();

  void IncrementAllDeltas(std::vector<double> next_errors);

  void CalculateAllGradients(size_t batch_size);

  void UpdateWeights(double learning_rate);

  std::vector<double> GetErrors() const;

  std::vector<Neuron> GetNeurons() const;

 private:
  const double kBias = 1;

  std::vector<Neuron> neurons_;
  std::vector<std::vector<double>>* weights_;
  std::vector<double> values_;
  std::vector<double> errors_;

  bool is_input_layer_ = false;
  bool is_output_layer_ = false;
};
}