//
// Created by ash on 12/3/20.
//

#include "visualizer/network_visualization.h"

#include "cinder/gl/gl.h"

using glm::vec2;
namespace neural_net {

namespace visualizer {

NetworkVisualization::NetworkVisualization(double height, double width,
                                           double margin)
    : kWindowHeight(height), kWindowWidth(width), kMargin(margin) {
  std::string img_dir =
      "/home/ash/UIUC/CS126/Cinder/my_projects/final-project-ash-maheriya/"
      "mnist/train/img/";
  std::string lbl_dir =
      "/home/ash/UIUC/CS126/Cinder/my_projects/final-project-ash-maheriya/"
      "mnist/train/lbl/";
  network_.LoadData(img_dir, lbl_dir);
  network_.Train();
}

size_t
neural_net::visualizer::NetworkVisualization::GetNumberOfNetworkLayers() {
  return network_.GetNumHiddenLayers() + 2;
}
void NetworkVisualization::Draw() {
  DrawInputLayer();
  DrawHiddenLayer();
  DrawOutputLayer();
}

void NetworkVisualization::DrawInputLayer() {
  ci::gl::color(ci::Color("red"));
  vec2 position(kMargin / 2, kMargin * 2);
  float num_neurons = network_.GetLayers()[0].GetNeurons().size();
  for (float i = 0; i < num_neurons; i++) {
    ci::gl::drawSolidCircle(position, 1);
    position.x += (kWindowWidth - kMargin) / num_neurons;
  }
}
void NetworkVisualization::DrawHiddenLayer() {
  ci::gl::color(ci::Color("red"));
  vec2 position(kMargin / 2, kWindowHeight / 2);
  float num_neurons = network_.GetLayers()[1].GetNeurons().size();
  for (float i = 0; i < num_neurons; i++) {
    ci::gl::drawSolidCircle(position, 4);
    position.x += (kWindowWidth - kMargin) / num_neurons;
  }
}
void NetworkVisualization::DrawOutputLayer() {
  ci::gl::color(ci::Color("red"));
  vec2 position(kWindowWidth/2, kWindowHeight - kMargin);
  float num_neurons = network_.GetLayers()[2].GetNeurons().size();
  for (float i = 0; i < num_neurons; i++) {
    ci::gl::drawSolidCircle(position, 10);
    position.x += (kWindowWidth - kMargin) / num_neurons;
  }
}

}  // namespace visualizer
}  // namespace neural_net