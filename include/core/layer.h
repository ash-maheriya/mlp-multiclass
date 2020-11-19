//
// Created by ash on 11/16/20.
//
#pragma once

#include <vector>
namespace neural_net {
class Layer {
 public:
  Layer(size_t size, size_t next_layer_size);
  void ForwardPass();
  double GetSize() const;

 private:
  std::vector<std::vector<double>> weights_;
  std::vector<double> values_;
  const size_t kSize;
  const double kBias = 1;
};
}