#include <catch2/catch.hpp>
#include "core/network.h"
#include "core/network.h"
#include "core/layer.h"
#include "core/neuron.h"

using neural_net::Neuron;
TEST_CASE("Forward Pass is accurate") {
  Neuron neuron;
  std::vector<float> weights;
  weights.push_back(0.6);
  std::vector<float> values;
  values.push_back(0.4);
  REQUIRE(neuron.ForwardPass(weights, values) == Approx(0.55971f));
}