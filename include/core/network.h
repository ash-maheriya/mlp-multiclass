//
// Created by ash on 11/14/20.
//
#pragma once
#include "common_types.h"
#include "layer.h"

namespace neural_net {

class Network {
 public:
  // TODO: READ NETWORK DEFINITION FROM JSON FILE
  Network(size_t image_size);

  /**
   * Runs the training functionality through every layer
   */
  void Train();

  /**
   * Returns the number of hidden layers in the network
   * @return the number of hidden layers in the network
   */
  size_t GetNumHiddenLayers();

  void BackPropagation(std::vector<size_t> labels);

  void LoadImages();

  double GetSparseCategoricalCrossEntropy(double output_activation, size_t ground_truth);
 private:
  const size_t kImageSize;

  Weight_Collection_t weights_;
  size_t num_hidden_layers_ = 1;
  std::vector<Layer> layers_;
  double learning_rate_ = 0.1;
  std::vector<Image_t> images_;
  std::vector<std::vector<size_t>> labels_;
};
} // namespace neural_net