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
  Network();
  // back propagation
  size_t GetNumHiddenLayers();
 private:
  Weight_Collection_t weights_;
  size_t num_hidden_layers_ = 2;
  std::vector<Layer> layers_;
};
} // namespace neural_net