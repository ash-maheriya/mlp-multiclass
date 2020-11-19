//
// Created by ash on 11/14/20.
//

#include "../../include/core/network.h"

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
  weights_[0] = vector<vector<double>>(28*28, vector<double>(10)); // input layer for 28*28 images
  weights_[1] = vector<vector<double>>(10, vector<double>(10));
  weights_[2] = vector<vector<double>>(10, vector<double>(10));
  weights_[3] = vector<vector<double>>(10, vector<double>(10));
  for (int layer = 0; layer < num_hidden_layers_+2; layer++) {
    for (int i = 0; i < weights_[layer].size(); i++) {
      for (int j = 0; j < weights_[layer][i].size(); i++) {
        weights_[layer][i][j] = rand() % 10;
      }
    }
  }

  // Creating the layers
  for (int i = 0; i < num_hidden_layers_+2; i++) {
    layers_.push_back(Layer(&weights_[i]));
  }I
}
size_t Network::GetNumHiddenLayers() {
  return num_hidden_layers_;
}
} // namespace neural_net