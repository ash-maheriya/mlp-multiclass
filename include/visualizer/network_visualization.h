//
// Created by ash on 12/3/20.
//
#pragma once

#include "core/network.h"
#include "cinder/gl/gl.h"

namespace neural_net {

namespace visualizer {

class NetworkVisualization {
 public:
  NetworkVisualization(double height, double width, double margin);

  void Draw();

  size_t GetNumberOfNetworkLayers();

  const size_t kImageSize = 28;

 private:
  void PlotInputLayer();

  void PlotHiddenLayer();

  void PlotOutputLayer();

  void DrawNetwork();

  const float kWindowHeight;

  const float kWindowWidth;

  const float kMargin;

  std::vector<size_t> neuron_sizes;

  Network network_ = Network(kImageSize);

  std::vector<std::vector<glm::vec2>> neuron_positions_;
};
}  // namespace visualizer
}  // namespace neural_net
