#include <iostream>
#include <ostream>

#include "core/network.h"

using neural_net::Network;

int main(int argc, char** argv) {

  Network network(28);
  std::string img_dir = "/home/ash/UIUC/CS126/Cinder/my_projects/final-project-ash-maheriya/mnist/train/img/";
  std::string lbl_dir = "/home/ash/UIUC/CS126/Cinder/my_projects/final-project-ash-maheriya/mnist/train/lbl/";
  std::cout << "Beginning to load data" << std::endl;
  network.LoadData(img_dir, lbl_dir);
  std::cout << "Finished loading, beginning training" << std::endl;
  network.Train();
  std::cout << "Finished training!" << std::endl;
  return 0;
}