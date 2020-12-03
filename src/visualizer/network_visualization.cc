//
// Created by ash on 12/3/20.
//

#include "visualizer/network_visualization.h"

using glm::vec2;
using std::vector;
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

  neuron_sizes.push_back(1);
  neuron_sizes.push_back(4);
  neuron_sizes.push_back(10);
}

size_t
neural_net::visualizer::NetworkVisualization::GetNumberOfNetworkLayers() {
  return network_.GetNumHiddenLayers() + 2;
}
void NetworkVisualization::Draw() {
  neuron_positions_.clear();
  for (size_t i = 0; i < network_.GetNumHiddenLayers()+2; i++) {
    neuron_positions_.push_back(vector<vec2>());
  }
  PlotInputLayer();
  PlotHiddenLayer();
  PlotOutputLayer();
  DrawNetwork();
}

void NetworkVisualization::PlotInputLayer() {
  ci::gl::color(ci::Color("red"));
  vec2 position(kMargin / 4, kMargin * 2);
  float num_neurons = network_.GetLayers()[0].GetNeurons().size();
  for (float i = 0; i < num_neurons; i++) {
    neuron_positions_[0].push_back(position);
    position.x += (kWindowWidth - kMargin / 2) / num_neurons;
  }
}
void NetworkVisualization::PlotHiddenLayer() {
  ci::gl::color(ci::Color("red"));
  vec2 position(kMargin / 4, kWindowHeight / 2);
  float num_neurons = network_.GetLayers()[1].GetNeurons().size();
  for (float i = 0; i < num_neurons; i++) {
    neuron_positions_[1].push_back(position);
    position.x += (kWindowWidth - kMargin / 2) / num_neurons;
  }
}
void NetworkVisualization::PlotOutputLayer() {
  ci::gl::color(ci::Color("red"));
  vec2 position(kWindowWidth / 2, kWindowHeight - kMargin);
  float num_neurons = network_.GetLayers()[2].GetNeurons().size();
  for (float i = 0; i < num_neurons; i++) {
    neuron_positions_[2].push_back(position);
    position.x += (kWindowWidth - kMargin / 2) / num_neurons;
  }
}

void NetworkVisualization::DrawNetwork() {
  for (size_t layer = 1; layer < network_.GetNumHiddenLayers()+2; layer++) {
    for (const vec2& first_position : neuron_positions_[layer]) {
      for (const vec2& second_position : neuron_positions_[layer-1]) {
        ci::gl::color(ci::Color("grey"));
        ci::gl::drawLine(first_position, second_position);
        ci::gl::color(ci::Color("red"));
        ci::gl::drawSolidCircle(first_position, neuron_sizes[layer]);
        ci::gl::drawSolidCircle(second_position, neuron_sizes[layer-1]);
      }
    }
  }
}

}  // namespace visualizer
}  // namespace neural_net