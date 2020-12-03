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

  /**
   * Returns how many neurons are in the layer
   * @return how many neurons are in the layer
   */
  float GetSize() const;

  void CalculateErrors(std::vector<float> next_errors);

  void CalculateOutputError(size_t label);

  void UpdateValues();

  void IncrementAllDeltas(std::vector<float> next_errors);

  void CalculateAllGradients(size_t batch_size);

  void UpdateWeights(float learning_rate);

  std::vector<float> GetErrors() const;

  std::vector<Neuron> GetNeurons() const;

  void ResetAllDeltas();

 private:
  const float kBias = 1;

  std::vector<Neuron> neurons_;
  std::vector<std::vector<float>> weights_;
  std::vector<float> values_;
  std::vector<float> errors_;
};
}