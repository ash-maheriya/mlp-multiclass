//
// Created by ash on 11/14/20.
//

#include "../../include/core/network.h"

#include <tgmath.h>

#include <iostream>
using std::vector;
namespace neural_net {

Network::Network(size_t image_size) : kImageSize(image_size){
  // Creating a new random seed the program is run
  time_t seconds;
  seconds = time(nullptr);
  srand(static_cast<int>(seconds));

  // Initializing and randomizing the weights
  weights_ = Weight_Collection_t(num_hidden_layers_+2);
  std::cout << weights_.size();
  weights_[0] = vector<vector<double>>(kImageSize*kImageSize + 1, vector<double>(10)); // input layer for 28*28 images
  weights_[1] = vector<vector<double>>(256, vector<double>(1)); // first hidden layer
//  weights_[2] = vector<vector<double>>(10 + 1, vector<double>(10)); // second hidden layer
  weights_[2] = vector<vector<double>>(1, vector<double>(1));
//  weights_[3] = vector<vector<double>>(10 + 1, vector<double>(10)); // output layer
  for (size_t layer = 0; layer < num_hidden_layers_+2; layer++) {
    for (size_t i = 0; i < weights_[layer].size(); i++) {
      weights_[layer][i][0] = 1;
      for (size_t j = 1; j < weights_[layer][i].size(); j++) {
        weights_[layer][i][j] = rand() % 10;
      }
    }
  }

  // Creating the layers
  layers_.push_back(Layer(&weights_[0], true, false));
  for (size_t i = 1; i < num_hidden_layers_+1; i++) {
    layers_.push_back(Layer(&weights_[i], false, false));
  }
  layers_.push_back(Layer(&weights_[weights_.size()-1], false, true));
}

size_t Network::GetNumHiddenLayers() {
  return num_hidden_layers_;
}

void Network::Train() {
  for (int i = 0; i < images_.size(); i++) {
    layers_[1].ForwardPassHidden(&layers_[0], &layers_[2]);
    double cost = GetSparseCategoricalCrossEntropy(
        layers_[2].ForwardPassOutput(&layers_[1]), 1);
    BackPropagation(labels_[i]);
    layers_[num_hidden_layers_+1].CalculateAllGradients(images_.size());
    layers_[num_hidden_layers_].CalculateAllGradients(images_.size());
  }
}

double Network::GetSparseCategoricalCrossEntropy(double output_activation, size_t ground_truth) {
  // dot product of the negative labels and the log of the prediction
  // since it's one-hot (only one label per image), it's basically just the
  // negative log (base 10) of the predicted value of the correct class
  if (ground_truth == 1) {
    return -1.0 * (log10(output_activation));
  } else {
    return -1.0 * (log10(1.0 - output_activation));
  }
}

void Network::BackPropagation(std::vector<size_t> labels) {
  layers_[num_hidden_layers_+1].CalculateOutputError(labels);
  layers_[num_hidden_layers_].CalculateErrors(layers_[num_hidden_layers_+1].GetErrors());
  layers_[num_hidden_layers_+1].IncrementAllDeltas(layers_[num_hidden_layers_+1].GetErrors());
  layers_[num_hidden_layers_].IncrementAllDeltas(layers_[num_hidden_layers_+1].GetErrors());
}
} // namespace neural_net