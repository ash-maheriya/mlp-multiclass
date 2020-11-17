//
// Created by ash on 11/16/20.
//

#ifndef FINAL_PROJECT_LAYER_H
#define FINAL_PROJECT_LAYER_H

#include <vector>
class Layer {
 public:
  Layer(size_t size);
  std::vector<double> weights;
  std::vector<double> values;
 private:
  const size_t kSize;
  const double kBias = 1;
};

#endif  // FINAL_PROJECT_LAYER_H
