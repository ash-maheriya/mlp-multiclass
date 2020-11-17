//
// Created by ash on 11/16/20.
//

#include "core/layer.h"
Layer::Layer(size_t size) : kSize(size){
  for (size_t i = 0; i <= size; i++) {
    weights.push_back(0);
    values.push_back(0);
  }
  values[0] = 1;
  weights[0] = kBias;
}
double Layer::GetSize() const{
  return kSize;
}
