//
// Created by ash on 11/16/20.
//

#include "core/layer.h"
#include <vector>
using std::vector;
namespace neural_net {
Layer::Layer(std::vector<std::vector<double>>* weights, bool is_input, bool is_output) : weights_(weights), is_input_layer_(is_input), is_output_layer_(is_output){
  values_.push_back(kBias);
  for (size_t i = 0; i < weights_->size()-1; i++) {
    neurons_.push_back(Neuron());
    values_.push_back(0);
  }
}

double Layer::GetSize() const{
  return neurons_.size();
}
void Layer::UpdateValues() {
  for (size_t i = 1; i < values_.size(); i++) {
    values_[i] = neurons_[i].GetActivation();
  }
}
void Layer::ForwardPassHidden(Layer* prev_layer, Layer* next_layer) {
  vector<double> weights;
  for (size_t i = 0; i < neurons_.size(); i++) {
    for (size_t j = 0; j < prev_layer->neurons_.size(); j++) {
      weights.push_back(prev_layer->weights_->at(j).at(i));
    }
    neurons_[i].ForwardPass(weights, prev_layer->values_);
  }
}

void Layer::ForwardPassInput(Layer* next_layer) {

}

void Layer::ForwardPassOutput(Layer* prev_layer) {
  vector<double> weights;
  for (size_t i = 0; i < neurons_.size(); i++) {
    for (size_t j = 0; j < prev_layer->neurons_.size(); j++) {
      weights.push_back(prev_layer->weights_->at(j).at(i));
    }
    neurons_[i].ForwardPass(weights, prev_layer->values_);
  }
}
}
// namespace neural_net