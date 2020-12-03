//
// Created by ash on 11/14/20.
//

#include "../../include/core/network.h"

#include <tgmath.h>

#include <cstring>
#include <fstream>
#include <iostream>

#include "dirent.h"
using std::vector;
using std::ifstream;
using std::istream;
namespace neural_net {

Network::Network(size_t image_size) : kImageSize(image_size){
  // Creating a new random seed the program is run
  time_t seconds;
  seconds = time(nullptr);
  srand(static_cast<int>(seconds));

  // Initializing and randomizing the weights
  weights_ = Weight_Collection_t(num_hidden_layers_+2);
  std::cout << weights_.size();
  weights_[1] = vector<vector<float>>(256, vector<float>(28*28+1)); // input layer for 28*28 images
  weights_[2] = vector<vector<float>>(1, vector<float>(257)); // first hidden layer
//  weights_[2] = vector<vector<float>>(10 + 1, vector<float>(10)); // second hidden layer
//  weights_[3] = vector<vector<float>>(10 + 1, vector<float>(10)); // output layer
  for (size_t layer = 1; layer < num_hidden_layers_+2; layer++) {
    for (size_t i = 0; i < weights_[layer].size(); i++) {
      weights_[layer][i][0] = 1;
      for (size_t j = 1; j < weights_[layer][i].size(); j++) {
        //weights_[layer][i][j] = rand() % 10;
        weights_[layer][i][j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      }
    }
  }

  // Creating the layers
  layers_.push_back(Layer(vector<vector<float>>(0)));
  for (size_t i = 1; i < num_hidden_layers_+1; i++) {
    layers_.push_back(Layer(weights_[i]));
  }
  layers_.push_back(Layer(weights_[weights_.size()-1]));
}

size_t Network::GetNumHiddenLayers() {
  return num_hidden_layers_;
}

void Network::Train() {
  std::cout << "Number of images: "  << images_.size() << std::endl;
  std::cout << "Number of labels: " << labels_.size() << std::endl;
  //for (size_t i = 0; i < images_.size(); i++) {
  for (size_t i = 0; i < 50; i++) {
    layers_[0].LoadInputActivations(images_[i]);
    layers_[1].ForwardPassHidden(layers_[0]);
    //float cost = GetSparseCategoricalCrossEntropy(
    //    layers_[2].ForwardPassOutput(&layers_[1]), 1);
    layers_[2].ForwardPassOutput(layers_[1]);
    BackPropagation(labels_[i]);
    layers_[num_hidden_layers_+1].CalculateAllGradients(images_.size());
    layers_[num_hidden_layers_].CalculateAllGradients(images_.size());
  }
  for (size_t i = 1; i < layers_.size(); i++) {
    layers_[i].UpdateWeights(learning_rate_);
  }
}

float Network::GetSparseCategoricalCrossEntropy(float output_activation, size_t ground_truth) {
  if (ground_truth == 1) {
    return -1.0 * (log10(output_activation));
  } else {
    return -1.0 * (log10(1.0 - output_activation));
  }
}

void Network::BackPropagation(size_t label) {
  layers_[num_hidden_layers_+1].CalculateOutputError(label);
  layers_[num_hidden_layers_].CalculateErrors(layers_[num_hidden_layers_+1].GetErrors());
  layers_[num_hidden_layers_+1].IncrementAllDeltas(layers_[num_hidden_layers_+1].GetErrors());
  layers_[num_hidden_layers_].IncrementAllDeltas(layers_[num_hidden_layers_+1].GetErrors());
}

void Network::LoadData(std::string& images_dir, std::string& labels_dir) {
  images_.clear();
  labels_.clear();
  // code for iterating over directory from: https://stackoverflow.com/questions/612097/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-c
  DIR *img_dir;
  struct dirent *img_ent;
  size_t img_count = 0;
  if ((img_dir = opendir(images_dir.c_str())) != NULL) {
    while ((img_ent = readdir(img_dir)) != NULL) {
      std::string f_name = img_ent->d_name;
      if (!strcmp(f_name.c_str(), ".") || !strcmp(f_name.c_str(), "..")) {
        continue;
      }
      std::string img_file = images_dir + f_name;
      ifstream training_images;
      training_images.open(img_file, ifstream::in);
      float pixel;
      Image_t img = vector<vector<float>>(kImageSize, vector<float>(kImageSize));
      for (size_t row = 0; row < 28; row++) {
        for (size_t col = 0; col < 28; col++) {
          if (training_images.read(reinterpret_cast<char*>(&pixel), sizeof(pixel))) {
            img[row][col] = pixel/255.0;
          }
        }
      }
      images_.push_back(img);
      img_count++;
    }
  }

  DIR *lbl_dir;
  struct dirent *lbl_ent;
  size_t lbl_count = 0;
  if ((lbl_dir = opendir(labels_dir.c_str())) != NULL) {
    while ((lbl_ent = readdir(lbl_dir)) != NULL) {
      std::string f_name = lbl_ent->d_name;
      if (!strcmp(f_name.c_str(), ".") || !strcmp(f_name.c_str(), "..")) {
        continue;
      }
      std::string lbl_file = labels_dir + f_name;
      ifstream training_labels;
      training_labels.open(lbl_file, ifstream::in);
      u_char label;
      if (training_labels.read(reinterpret_cast<char*>(&label), sizeof(label))) {
        labels_.push_back((size_t)label);
        lbl_count++;
      }
    }
  }
}

// loading images
std::istream& operator>>(std::istream& is, Network& network) {

  return is;
}

} // namespace neural_net