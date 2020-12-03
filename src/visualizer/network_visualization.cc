//
// Created by ash on 12/3/20.
//

#include "visualizer/network_visualization.h"

namespace neural_net {

namespace visualizer {

NetworkVisualization::NetworkVisualization() {
  std::string img_dir = "/home/ash/UIUC/CS126/Cinder/my_projects/final-project-ash-maheriya/mnist/train/img/";
  std::string lbl_dir = "/home/ash/UIUC/CS126/Cinder/my_projects/final-project-ash-maheriya/mnist/train/lbl/";
  network_.LoadData(img_dir, lbl_dir);
  network_.Train();}

size_t neural_net::visualizer::NetworkVisualization::GetNumberOfNetworkLayers() {
  return network_.GetNumHiddenLayers() + 2;
}

}  // namespace visualizer
}  // namespace neural_net