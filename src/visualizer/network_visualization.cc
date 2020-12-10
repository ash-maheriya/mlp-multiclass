//
// Created by ash on 12/3/20.
//

#include "visualizer/network_visualization.h"
#include <cmath>

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
  network_.LoadTrainingData(img_dir, lbl_dir);
  std::string load_file = "/home/ash/UIUC/CS126/Cinder/my_projects/final-project-ash-maheriya/include/core/model.bin";
  network_.LoadNetwork(load_file);
  neuron_sizes.push_back(1);  // input neuron size
  neuron_sizes.push_back(10);  // hidden neuron size
  neuron_sizes.push_back(20); // output neuron size
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
  ci::gl::color(ci::Color("red"));
  PlotInputLayer();
  PlotHiddenLayer();
  PlotOutputLayer();
  DrawNetwork();
}

void NetworkVisualization::PlotInputLayer() {
  vec2 position(kMargin / 4, kMargin * 2);
  float num_neurons = network_.GetLayers()[0].GetNeurons().size();
  for (float i = 0; i < num_neurons; i++) {
    neuron_positions_[0].push_back(position);
    position.x += (kWindowWidth - kMargin / 2) / num_neurons;
  }
}
void NetworkVisualization::PlotHiddenLayer() {
  vec2 position(kMargin / 4, kWindowHeight / 2);
  float num_neurons = network_.GetLayers()[1].GetNeurons().size();
  for (float i = 0; i < num_neurons; i++) {
    neuron_positions_[1].push_back(position);
    position.x += (kWindowWidth - kMargin / 2) / num_neurons;
  }
}
void NetworkVisualization::PlotOutputLayer() {
  vec2 position(kWindowWidth / 2, kWindowHeight - kMargin);
  float num_neurons = network_.GetLayers()[2].GetNeurons().size();
  for (float i = 0; i < num_neurons; i++) {
    neuron_positions_[2].push_back(position);
    position.x += (kWindowWidth - kMargin / 2) / num_neurons;
  }
}

void NetworkVisualization::DrawNetwork() {
  for (size_t layer = 1; layer < network_.GetNumHiddenLayers()+2; layer++) {
    for (size_t i = 0; i < neuron_positions_[layer].size(); i++) {
      for (size_t j = 0; j < neuron_positions_[layer-1].size(); j++) {
        float value = network_.GetLayers()[layer].GetWeights()[i][j];
        ci::gl::color(ci::Color(255*std::abs(value), 100*std::abs(value), 100*std::abs(value)));
        ci::gl::drawLine(neuron_positions_[layer][i], neuron_positions_[layer-1][j]);
        ci::gl::color(ci::Color("red"));
        ci::gl::drawSolidCircle(neuron_positions_[layer][i], neuron_sizes[layer]);
        ci::gl::drawSolidCircle(neuron_positions_[layer-1][j], neuron_sizes[layer-1]);
      }
    }
  }
}

}  // namespace visualizer
}  // namespace neural_net