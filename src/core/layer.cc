//
// Created by ash on 11/16/20.
//

#include "core/layer.h"

#include <vector>
using std::vector;
namespace neural_net {
Layer::Layer(vector<vector<float>> weights) : weights_(weights){
  values_.push_back(1);
  errors_.push_back(1);
  if (weights_.size() == 0) {
    for (size_t j = 0; j < 28*28; j++) {
      neurons_.push_back(Neuron());
      values_.push_back(0);
    }
  } else {
    for (size_t i = 0; i < weights_.size(); i++) {
      neurons_.push_back(Neuron());
      values_.push_back(0);
      errors_.push_back(0);
    }
    deltas_ = Delta_Collection_t(weights_.size(), vector<float>(weights_[0].size()));
    gradients_ = Delta_Collection_t(weights_.size(), vector<float>(weights_[0].size()));
  }
}

void Layer::UpdateValues() {
  for (size_t i = 1; i < values_.size(); i++) {
    values_[i] = neurons_[i-1].GetActivation();
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

void Layer::CalculateErrors(const std::vector<std::vector<float>>& next_weights, const Error_Collection_t& next_errors) {
  vector<float> sigmoid_primes;
  for (const Neuron& neuron : neurons_) {
    sigmoid_primes.push_back(neuron.GetActivation() * (1.0 - neuron.GetActivation()));
  }

  vector<float> weight_error_products;
  float value;
  for (size_t i = 1; i < next_weights[0].size(); i++) { // iterating through current layer's neurons
    value = 0;
    for (size_t j = 0; j < next_weights.size(); j++) {  // iterating through next layer's neurons
      value += next_weights[j][i] * next_errors[j];
    }
    weight_error_products.push_back(value);
  }

  for (size_t i = 0; i < errors_.size(); i++) {
    errors_[i] = weight_error_products[i] * sigmoid_primes[i];
  }
}

std::vector<Neuron> Layer::GetNeurons() const{
  return neurons_;
}

void Layer::CalculateOutputError(size_t label) {
  errors_[0] = neurons_[0].GetActivation() - label;
}

Error_Collection_t Layer::GetErrors() const {
  return errors_;
}

void Layer::IncrementAllDeltas(const std::vector<float>& prev_values) {
  for (size_t i = 0; i < deltas_.size(); i++) {
    deltas_[i][0] += errors_[i];
    for (size_t j = 1; j < deltas_[i].size(); j++) {
      deltas_[i][j] += prev_values[j] * errors_[i];
    }
  }
}

void Layer::CalculateAllGradients(size_t batch_size) {
  for (size_t i = 0; i < gradients_.size(); i++) {
    for (size_t j = 0; j < gradients_[i].size(); j++) {
      gradients_[i][j] = deltas_[i][j] / batch_size;
    }
  }
}

void Layer::UpdateWeights(float learning_rate) {
  for (size_t i = 0; i < weights_.size(); i++) {
    for (size_t j = 0; j < weights_[i].size(); j++) {
      weights_[i][j] -= learning_rate * gradients_[i][j];
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
  for (size_t i = 0; i < gradients_.size(); i++) {
    for (size_t j = 0; j < gradients_[i].size(); j++) {
      deltas_[i][j] = 0;
    }
  }
}

std::vector<std::vector<float>> Layer::GetWeights() const {
  return weights_;
}

void Layer::SetWeight(size_t neuron_index, size_t weight_index, float value) {
  weights_[neuron_index][weight_index] = value;
}
std::vector<float> Layer::GetValues() {
  return values_;
}

}
// namespace neural_net