//
// Created by ash on 12/3/20.
//
#pragma once

#include "core/network.h"

namespace neural_net {

namespace visualizer {

class NetworkVisualization {
 public:
  NetworkVisualization(double height, double width, double margin);

  void Draw();

  size_t GetNumberOfNetworkLayers();

  const size_t kImageSize = 28;

 private:
  void DrawInputLayer();

  void DrawHiddenLayer();

  void DrawOutputLayer();

  const float kWindowHeight;

  const float kWindowWidth;

  const float kMargin;

  Network network_ = Network(kImageSize);
};
}  // namespace visualizer
}  // namespace neural_net
