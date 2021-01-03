#include <catch2/catch.hpp>
#include "core/network.h"
#include "core/layer.h"
#include "core/neuron.h"

using neural_net::Network;
TEST_CASE("Loss function is accurate") {
  Network network(28);
  std::string load_file = "/home/ash/UIUC/CS126/Cinder/my_projects/final-project-ash-maheriya/include/core/model.bin";
  network.LoadNetwork(load_file);
  std::vector<float> outputs;
  outputs.push_back(0.6);
  outputs.push_back(0.4);
  SECTION("Ground truth is 1") {
    REQUIRE(network.CalculateLoss(outputs, 0) == Approx(0.510826));
  }
  SECTION("Ground truth is 0") {
    REQUIRE(network.CalculateLoss(outputs, 1) == Approx(0.916291));
  }
}