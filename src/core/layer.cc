//
// Created by ash on 11/16/20.
//

#include "core/layer.h"
#include <vector>
namespace neural_net {
Layer::Layer(std::vector<std::vector<double>>* weights) : weights_(weights){

}
double Layer::GetSize() const{
  return weights_->size();
}
} // namespace neural_net