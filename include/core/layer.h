//
// Created by ash on 11/16/20.
//
#pragma once

#include <vector>
namespace neural_net {
class Layer {
 public:
  Layer(size_t size);
  std::vector<double> weights;
  std::vector<double> values;
  double GetSize() const;

 private:
  const size_t kSize;
  const double kBias = 1;
};
}