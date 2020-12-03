//
// Created by ash on 11/16/20.
//

#include "core/layer.h"

#include <iostream>
#include <vector>
using std::vector;
namespace neural_net {
Layer::Layer(vector<vector<float>> weights) : weights_(weights){
  values_.push_back(kBias);
  errors_.push_back(1);
  if (weights_.size() == 0) {
    for (size_t j = 0; j < 28*28; j++) {
      neurons_.push_back(Neuron());
      values_.push_back(0);
      errors_.push_back(0);
    }
  }
  for (size_t i = 0; i < weights_.size(); i++) {
    neurons_.push_back(Neuron());
    values_.push_back(0);
    errors_.push_back(0);
  }
}

float Layer::GetSize() const{
  return neurons_.size();
}
void Layer::UpdateValues() {
  for (size_t i = 1; i < values_.size(); i++) {
    values_[i] = neurons_[i].GetActivation();
    errors_[i] = neurons_[i].GetError();
  }
}
void Layer::ForwardPassHidden(Layer& prev_layer) {
  for (size_t i = 0; i < neurons_.size(); i++) {
    neurons_[i].ForwardPass(weights_[i], prev_layer.values_);
  }
  UpdateValues();
}


float Layer::ForwardPassOutput(Layer& prev_layer) {
  float output = neurons_[0].ForwardPass(weights_[0], prev_layer.values_);
  UpdateValues();
  return output;
}

void Layer::CalculateErrors(std::vector<float> next_errors) {
  float value;
  for (size_t i = 0; i < neurons_.size(); i++) {
    value = 0;
    for (size_t j = 1; j < weights_[i].size(); j++) {
      value += weights_[i][j] * next_errors[j];
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
std::vector<float> Layer::GetErrors() const {
  return errors_;
}

void Layer::IncrementAllDeltas(std::vector<float> next_errors) {
  for (size_t i = 0; i < neurons_.size(); i++) {
    neurons_[i].IncrementDelta(next_errors[i]);
  }
}
void Layer::CalculateAllGradients(size_t batch_size) {
  for (Neuron& neuron : neurons_) {
    neuron.CalculateGradient(batch_size);
  }
}
void Layer::UpdateWeights(float learning_rate) {
  for (size_t i = 1; i < weights_.size(); i++) {
    for (size_t j = 0; j < weights_[i].size(); j++) {
      weights_[i][j] -= learning_rate * neurons_[i-1].GetGradient();
    }
  }
}

void Layer::LoadInputActivations(const Image_t& img) {
  size_t index = 0;
  for (size_t row = 0; row < img.size(); row++) {
    for (size_t col = 0; col < img[row].size(); col++) {
      neurons_[index].SetActivation(img[row][col]);
      index++;
    }
  }
  UpdateValues();
}
void Layer::ResetAllDeltas() {
  for (Neuron& neuron : neurons_) {
    neuron.ResetDelta();
  }
}

}
// namespace neural_net