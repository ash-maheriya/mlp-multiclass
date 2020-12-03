#include <visualizer/neural_network_app.h>

namespace neural_net {

using glm::vec2;
namespace visualizer {
ci::Font font = ci::Font("Suruma", 50);
NeuralNetworkApp::NeuralNetworkApp() : visualization_(kWindowHeight, kWindowWidth, kMargin){
  ci::app::setWindowSize(static_cast<int>(kWindowWidth),
                         static_cast<int>(kWindowHeight));
}

void NeuralNetworkApp::draw() {
  ci::Color8u background_color(0, 0, 0);  // black
  ci::gl::clear(background_color);

  visualization_.Draw();

  ci::gl::drawStringCentered(
      "This is a visual representation of the layers of the neural network.",
      glm::vec2(kWindowWidth / 2, kMargin / 2), ci::Color("white"), font);
  ci::gl::drawStringCentered(
      "Number of layers: " + std::to_string(visualization_.GetNumberOfNetworkLayers()),
      glm::vec2(kWindowWidth / 2, kMargin), ci::Color("white"), font);
}

void NeuralNetworkApp::keyDown(ci::app::KeyEvent event) {
  switch (event.getCode()) {
    case ci::app::KeyEvent::KEY_RETURN:
      break;
  }
}
}  // namespace visualizer

}  // namespace neural_net
