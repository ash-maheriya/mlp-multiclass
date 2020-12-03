//
// Created by ash on 12/3/20.
//
#pragma once

#include "core/network.h"

namespace neural_net {

namespace visualizer {

class NetworkVisualization {
 public:
  NetworkVisualization();

  void Draw();

  size_t GetNumberOfNetworkLayers();

 private:
  Network network_ = Network(28);
};
}  // namespace visualizer
}  // namespace neural_net
