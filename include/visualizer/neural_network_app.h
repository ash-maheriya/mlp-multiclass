#pragma once


#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

#include <core/network.h>
#include <visualizer/network_visualization.h>

namespace neural_net {

namespace visualizer {

/**
 * Displays a graph of a neural network
 */
class NeuralNetworkApp : public ci::app::App {
 public:
  NeuralNetworkApp();

  void draw() override;

  void keyDown(ci::app::KeyEvent event) override;

  const double kWindowHeight = 1900;

  const double kWindowWidth = 3750;

  const double kMargin = 150;
 private:
  NetworkVisualization visualization_;
};

}  // namespace visualizer

}  // namespace ideal_gas
