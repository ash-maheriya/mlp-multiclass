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
   * Runs the forward pass of every neuron in a hidden layer and passes the values
   * onto the next layer in the network
   * @param prev_layer the previous layer in the network
   */
  void ForwardPassHidden(Layer& prev_layer);

  /**
   * Assigns the pixel values of the image to the activation values of the
   * neurons in the layer
   * @param img given image to be examined
   */
  void LoadInputActivations(const Image_t& img);

  /**
   * Runs the forward pass of every neuron in the output layer
   * @param prev_layer the previous layer of the network
   * @return the computed activation of the final neuron
   */
  std::vector<float> ForwardPassOutput(Layer& prev_layer);

  /**
   * Calculates the error of each node in the network
   * @param next_weights the weights in the next layer of the network
   * @param next_errors the errors of the next layer of the network
   */
  void CalculateErrors(const std::vector<std::vector<float>>& next_weights, const Error_Collection_t& next_errors);

  /**
   * Calculates the error of the final layer of the network
   * @param label the ground truth that the output will be compared to
   */
  void CalculateOutputError(size_t label);

  /**
   * Updates the layer's values to match the values of its neurons
   */
  void UpdateValues();

  /**
   * Accumulates the delta values to accomodate for the layer's errors
   * @param prev_values the activation values of the previous layer
   */
  void IncrementAllDeltas(const std::vector<float>& prev_values);

  /**
   * Averages out the delta values depending on batch size
   * @param batch_size size of the current training batch
   */
  void CalculateAllGradients(size_t batch_size);

  /**
   * Shifts the layer's weights depending on the gradients
   * @param learning_rate learning rate of the network
   */
  void UpdateWeights(float learning_rate);

  /**
   * Sets all delta values to zero
   */
  void ResetAllDeltas();

  std::vector<std::vector<float>> GetWeights() const;

  void SetWeight(size_t neuron_index, size_t weight_index, float value);

  std::vector<float> GetValues();

  Error_Collection_t GetErrors() const;

  /**
   * Returns the layer's neurons
   * @return neurons
   */
  std::vector<Neuron> GetNeurons() const;

 private:
  std::vector<Neuron> neurons_;
  std::vector<std::vector<float>> weights_;
  std::vector<float> values_;
  Error_Collection_t errors_;
  Delta_Collection_t deltas_;
  Gradient_Collection_t gradients_;
};
}