#include <visualizer/neural_network_app.h>

using neural_net::visualizer::NeuralNetworkApp;

void prepareSettings(NeuralNetworkApp::Settings* settings) {
  settings->setResizable(false);
}

// This line is a macro that expands into an "int main()" function.
CINDER_APP(NeuralNetworkApp, ci::app::RendererGl, prepareSettings)
