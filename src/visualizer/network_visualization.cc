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
  std::string fashion_dir = "/home/ash/UIUC/CS126/Cinder/my_projects/final-project-ash-maheriya/fashion_mnist/train/img";
  network_.LoadData(img_dir, lbl_dir, fashion_dir);
  network_.Train();

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
//    for (const vec2& first_position : neuron_positions_[layer]) {
//      for (const vec2& second_position : neuron_positions_[layer-1]) {
//        // need something to get the weight value of the line being drawn so
//        // then the brightness of the color can be determined
//        ci::gl::color(ci::Color("gray"));
//        ci::gl::drawLine(first_position, second_position);
//        ci::gl::color(ci::Color("red"));
//        ci::gl::drawSolidCircle(first_position, neuron_sizes[layer]);
//        ci::gl::drawSolidCircle(second_position, neuron_sizes[layer-1]);
//      }
//    }
    for (size_t i = 0; i < neuron_positions_[layer].size(); i++) {
      for (size_t j = 0; j < neuron_positions_[layer-1].size(); j++) {
        float value = network_.GetLayers()[layer].GetWeights()[i][j];
        //ci::gl::color(ci::Color("gray"));
        ci::gl::color(ci::Color(255*std::abs(value), 0, 0));
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