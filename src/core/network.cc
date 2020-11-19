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
  weights_[0] = vector<vector<double>>(28*28 + 1, vector<double>(10)); // input layer for 28*28 images
  weights_[1] = vector<vector<double>>(10 + 1, vector<double>(10)); // first hidden layer
  weights_[2] = vector<vector<double>>(10 + 1, vector<double>(10)); // second hidden layer
  weights_[3] = vector<vector<double>>(10 + 1, vector<double>(10)); // output layer
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
void Network::ForwardPass() {
  layers_[0].ForwardPassInput(&layers_[1]);
  layers_[1].ForwardPassHidden(&layers_[0], &layers_[2]);
  layers_[2].ForwardPassHidden(&layers_[1], &layers_[3]);
  layers_[3].ForwardPassOutput(&layers_[2]);
}
} // namespace neural_net