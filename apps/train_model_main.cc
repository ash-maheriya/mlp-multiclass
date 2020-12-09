#include <iostream>
#include <ostream>

#include "core/network.h"

using neural_net::Network;

int main(int argc, char** argv) {

  Network network(28);

  std::string img_dir = "/home/ash/UIUC/CS126/Cinder/my_projects/final-project-ash-maheriya/mnist/train/img/";
  std::string lbl_dir = "/home/ash/UIUC/CS126/Cinder/my_projects/final-project-ash-maheriya/mnist/train/lbl/";
  std::string fashion_dir = "/home/ash/UIUC/CS126/Cinder/my_projects/final-project-ash-maheriya/fashion_mnist/train/img";

  std::string test_img_dir = "/home/ash/UIUC/CS126/Cinder/my_projects/final-project-ash-maheriya/mnist/train/img/";
  std::string test_lbl_dir = "/home/ash/UIUC/CS126/Cinder/my_projects/final-project-ash-maheriya/mnist/train/lbl/";
  std::string test_fashion_dir = "/home/ash/UIUC/CS126/Cinder/my_projects/final-project-ash-maheriya/fashion_mnist/train/img";
  std::cout << "Beginning to load data" << std::endl;
  network.LoadTrainingData(img_dir, lbl_dir, fashion_dir);
  network.LoadTestingData(test_img_dir, test_lbl_dir, test_fashion_dir);
  std::cout << "Finished loading, beginning training" << std::endl;
  //network.Train();
  std::string load_file = "/home/ash/UIUC/CS126/Cinder/my_projects/final-project-ash-maheriya/include/core/model.bin";
  network.LoadNetwork(load_file);
  std::cout << "Finished training!" << std::endl;
  network.ValidateNetwork();
  std::cout << "Finished validating!" << std::endl;
  return 0;
}