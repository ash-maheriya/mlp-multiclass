//
// Created by ash on 11/16/20.
//
#pragma once

#include <vector>
#include "neuron.h"
#include "common_types.h"
namespace neural_net {
/**
 * The layers that comprise the network
 */
class Layer {
 public:
  Layer(std::vector<std::vector<float>> weights);

  /**
   * Runs the forward pass of every neuron in the layer and passes the values
   * onto the next layer in the network
   * @param next_layer
   */
  void ForwardPassHidden(Layer& prev_layer);

  void LoadInputActivations(const Image_t& img);

  float ForwardPassOutput(Layer& prev_layer);

  void CalculateErrors(const std::vector<std::vector<float>>& next_weights, const Error_Collection_t& next_errors);

  void CalculateOutputError(size_t label);

  void UpdateValues();

  void IncrementAllDeltas(const std::vector<float>& prev_values);

  void CalculateAllGradients(size_t batch_size);

  void UpdateWeights(float learning_rate);

  Error_Collection_t GetErrors() const;

  std::vector<Neuron> GetNeurons() const;

  void ResetAllDeltas();

  std::vector<std::vector<float>> GetWeights() const;

  void SetWeight(size_t neuron_index, size_t weight_index, float value);

  std::vector<float> GetValues();

 private:
  std::vector<Neuron> neurons_;
  std::vector<std::vector<float>> weights_;
  std::vector<float> values_;
  Error_Collection_t errors_;
  Delta_Collection_t deltas_;
  Gradient_Collection_t gradients_;
};
}