//
// Created by ash on 11/14/20.
//

#include "../../include/core/network.h"
using std::vector;
namespace neural_net {

Network::Network(){
//  weights_ = Weight_Collection_t();
//  weights_[0] = vector<vector<double>>(28*28, vector<double>(10));
//  weights_[1] = vector<vector<double>>(10, vector<double>(10));
//  weights_[2] = vector<vector<double>>(10, vector<double>(10));
//  weights_[3] = vector<vector<double>>(10, vector<double>(10));
}
size_t Network::GetNumHiddenLayers() {
  return num_hidden_layers_;
}
} // namespace neural_net