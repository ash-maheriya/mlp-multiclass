//
// Created by ash on 11/16/20.
//

#include "core/layer.h"
#include <vector>
namespace neural_net {
Layer::Layer(size_t size, size_t next_layer_size) : kSize(size){
//  for (size_t i = 0; i <= size; i++) {
//    weights_.push_back(std::vector<double>);
//    values_.push_back(0);
//  }
//  values_[0] = 1;
//  weights_[0] = kBias;
}
double Layer::GetSize() const{
  return kSize;
}
} // namespace neural_net