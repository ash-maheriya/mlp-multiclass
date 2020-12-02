//
// Created by ash on 11/16/20.
//

#include "core/layer.h"
#include <vector>
using std::vector;
namespace neural_net {
Layer::Layer(std::vector<std::vector<double>>* weights, bool is_input, bool is_output) : weights_(weights), is_input_layer_(is_input), is_output_layer_(is_output){
  values_.push_back(kBias);
  errors_.push_back(1);
  for (size_t i = 0; i < weights_->size()-1; i++) {
    neurons_.push_back(Neuron());
    values_.push_back(0);
    errors_.push_back(0);
  }
}

double Layer::GetSize() const{
  return neurons_.size();
}
void Layer::UpdateValues() {
  for (size_t i = 1; i < values_.size(); i++) {
    values_[i] = neurons_[i].GetActivation();
    errors_[i] = neurons_[i].GetError();
  }
}
void Layer::ForwardPassHidden(Layer* prev_layer, Layer* next_layer) {
  vector<double> weights;
  weights.push_back(1);
  for (size_t i = 0; i < neurons_.size(); i++) {
    for (size_t j = 0; j < prev_layer->neurons_.size(); j++) {
      weights.push_back(prev_layer->weights_->at(j).at(i));
    }
    neurons_[i].ForwardPass(weights, prev_layer->values_);
  }
  UpdateValues();
}


double Layer::ForwardPassOutput(Layer* prev_layer) {
  vector<double> weights;
  weights.push_back(1);
  for (size_t j = 0; j < prev_layer->neurons_.size(); j++) {
    weights.push_back(prev_layer->weights_->at(j).at(0));
  }
  double output = neurons_[0].ForwardPass(weights, prev_layer->values_);
  UpdateValues();
  return output;
}

void Layer::CalculateErrors(std::vector<double> next_errors) {
  double value;
  for (size_t i = 1; i <= neurons_.size(); i++) {
    value = 0;
    for (size_t j = 0; j < weights_[i].size(); j++) {
      value += weights_->at(i)[j] * next_errors[j];
    }
    neurons_[i].CalculateError(value);
  }
  UpdateValues();
}
std::vector<Neuron> Layer::GetNeurons() const{
  return neurons_;
}
void Layer::CalculateOutputError(size_t label) {
  neurons_[0].SetError(neurons_[0].GetActivation() - label);
  UpdateValues();
}
std::vector<double> Layer::GetErrors() const {
  return errors_;
}

void Layer::IncrementAllDeltas(std::vector<double> next_errors) {
  for (size_t i = 0; i < neurons_.size(); i++) {
    neurons_[i].IncrementDelta(next_errors[i]);
  }
}
void Layer::CalculateAllGradients(size_t batch_size) {
  for (Neuron& neuron : neurons_) {
    neuron.CalculateGradient(batch_size);
  }
}
void Layer::UpdateWeights(double learning_rate) {
  for (size_t i = 1; i < weights_->size(); i++) {
    for (size_t j = 0; j < weights_[i].size(); j++) {
      weights_->at(i)[j] -= learning_rate * neurons_[i-1].GetGradient();
    }
  }
}

}
// namespace neural_net