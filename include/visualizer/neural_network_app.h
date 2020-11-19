#pragma once

#include <core/network.h>
#include <time.h>

#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

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

  const double kWindowWidth = 2500;

  const double kMargin = 150;

 private:
  Network network_;
};

}  // namespace visualizer

}  // namespace ideal_gas
