#include <catch2/catch.hpp>
#include "core/network.h"
#include "core/layer.h"
#include "core/neuron.h"

using neural_net::Network;
TEST_CASE("Loss function is accurate") {
  Network network(28);
  std::string load_file = "/home/ash/UIUC/CS126/Cinder/my_projects/final-project-ash-maheriya/include/core/model.bin";
  network.LoadNetwork(load_file);
  SECTION("Ground truth is 1") {
    REQUIRE(network.CalculateLoss(0.6, 1) == Approx(0.510826));
  }
  SECTION("Ground truth is 0") {
    REQUIRE(network.CalculateLoss(0.6, 0) == Approx(0.916291));
  }
}

TEST_CASE("Saving and loading a network to a file") {
  Network network_first(28);
  std::string img_dir = "/home/ash/UIUC/CS126/Cinder/my_projects/final-project-ash-maheriya/mnist/train/img/";
  std::string lbl_dir = "/home/ash/UIUC/CS126/Cinder/my_projects/final-project-ash-maheriya/mnist/train/lbl/";
  network_first.LoadTrainingData(img_dir, lbl_dir);
  network_first.Train();

  Network network_second(28);
  std::string load_file = "/home/ash/UIUC/CS126/Cinder/my_projects/final-project-ash-maheriya/include/core/model.bin";
  network_second.LoadNetwork(load_file);

  bool are_networks_equal = true;
  for (size_t layer = 1; layer < network_first.GetLayers().size(); layer++) {
    for (size_t i = 0; i < network_first.GetLayers()[layer].GetWeights().size(); i++) {
      for (size_t j = 0; j < network_first.GetLayers()[layer].GetWeights()[i].size(); j++) {
        if (network_first.GetLayers()[layer].GetWeights()[i][j] != Approx(network_second.GetLayers()[layer].GetWeights()[i][j])) {
          are_networks_equal = false;
        }
      }
    }
  }
  REQUIRE(are_networks_equal == true);
}
