//
// Created by ash on 11/14/20.
//

#include "../../include/core/network.h"

#include <tgmath.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

#include "dirent.h"
using std::ifstream;
using std::istream;
using std::vector;
namespace neural_net {

void PrintImage(Image_t img, size_t label) {
  for (size_t i = 0; i < img.size(); i++) {
    for (size_t j = 0; j < img[i].size(); j++) {
      if (img[i][j] == 0) {
        printf(" ");
      } else {
        printf("#");
      }
    }
    printf("\n");
  }
}

Network::Network(size_t image_size) : kImageSize(image_size) {
  // Creating a new random seed the program is run
  time_t seconds;
  seconds = time(nullptr);
  srand(static_cast<int>(seconds));

  // Initializing and randomizing the weights
  Weight_Collection_t weights = Weight_Collection_t(num_hidden_layers_ + 2);
  std::cout << weights.size();
  weights[1] = vector<vector<float>>(
      hidden_layer_size, vector<float>(kImageSize * kImageSize +
                                       1));  // input layer for 28*28 images
  weights[2] = vector<vector<float>>(
      kNumClasses, vector<float>(hidden_layer_size + 1));  // first hidden layer
  for (size_t layer = 1; layer < num_hidden_layers_ + 2; layer++) {
    for (size_t i = 0; i < weights[layer].size(); i++) {
      weights[layer][i][0] = 0;  // bias term
      for (size_t j = 1; j < weights[layer][i].size(); j++) {
        weights[layer][i][j] =
            static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5;
      }
    }
  }

  // Creating the layers
  layers_.push_back(Layer(vector<vector<float>>(0)));
  for (size_t i = 1; i < num_hidden_layers_ + 1; i++) {
    layers_.push_back(Layer(weights[i]));
  }
  layers_.push_back(Layer(weights[weights.size() - 1]));
}

size_t Network::GetNumHiddenLayers() {
  return num_hidden_layers_;
}

void Network::Train() {
  std::cout << "Number of images: " << images_.size() << std::endl;
  std::cout << "Number of labels: " << labels_.size() << std::endl;
  if (images_.empty() || labels_.empty()) {
    throw std::invalid_argument("Must load data before training");
  }

  float cost;
  size_t iterations = 0;
  size_t num_epochs = 40;
  bool is_epoch_done;
  size_t num_batches = images_.size() / batch_size;

  for (size_t epoch = 0; epoch < num_epochs; epoch++) {
    is_epoch_done = false;
    size_t image_index = 0;
    for (size_t i = 0; i < num_batches; i++) {  // looping through batches
      iterations++;
      cost = 0;
      layers_[1].ResetAllDeltas();
      layers_[2].ResetAllDeltas();
      for (size_t j = 0; j < batch_size; j++) {  // looping through images
        if (image_index > images_.size()) {
          is_epoch_done = true;
          break;
        }
        layers_[0].LoadInputActivations(images_[indices_[image_index]]);
        layers_[1].ForwardPassHidden(layers_[0]);
        std::vector<float> outputs = layers_[2].ForwardPassOutput(layers_[1]);
        cost += CalculateLoss(outputs, labels_[indices_[image_index]]);
        BackPropagation(labels_[indices_[image_index]]);
        image_index++;
      }
      if (is_epoch_done) {
        break;
      }

      for (size_t k = 1; k < layers_.size(); k++) {
        layers_[k].CalculateAllGradients(batch_size);
        layers_[k].UpdateWeights(learning_rate_);
      }
      if (iterations % 50 == 0) {
        std::cout << "Iteration: " << iterations
                  << ", Cost: " << cost / (batch_size) << ", Epochs: " << epoch + 1
                  << " out of " << num_epochs << std::endl;
        ValidateNetwork();
      }
    }
    std::string save_file =
        "/home/ash/UIUC/CS126/Cinder/my_projects/final-project-ash-maheriya/"
        "include/core/model.bin";
    SaveNetwork(save_file);
    std::random_shuffle(indices_.begin(), indices_.end());
  }
}

float Network::CalculateLoss(std::vector<float> output_activations, size_t ground_truth) {
//  if (ground_truth == 1) {
//    return -1.0 * (log(output_activation));
//  } else {
//    return -1.0 * (log(1.0 - output_activation));
//  }
  return -1.0 * log(output_activations[ground_truth]);
}

void Network::BackPropagation(size_t label) {
  // Calculate the errors for every layer except input
  layers_[num_hidden_layers_ + 1].CalculateOutputError(label);
  layers_[num_hidden_layers_].CalculateErrors(
      layers_[num_hidden_layers_ + 1].GetWeights(),
      layers_[num_hidden_layers_ + 1].GetErrors());

  // Accumulate the deltas for every layer except input
  layers_[num_hidden_layers_ + 1].IncrementAllDeltas(
      layers_[num_hidden_layers_].GetValues());
  layers_[num_hidden_layers_].IncrementAllDeltas(layers_[0].GetValues());
}

void Network::LoadTrainingData(std::string& images_dir,
                               std::string& labels_dir) {
  images_.clear();
  labels_.clear();
  // code for iterating over directory from:
  // https://stackoverflow.com/questions/612097/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-c
  DIR* lbl_dir;
  struct dirent* lbl_ent;
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
      if (training_labels.read(reinterpret_cast<char*>(&label),
                               sizeof(label))) {
//        if ((size_t)label == kPositiveClass) {
//          labels_.push_back(1);
//        } else {
//          labels_.push_back(0);
//        }
        labels_.push_back(size_t(label));
        lbl_count++;
      }
    }
  }

  DIR* img_dir;
  struct dirent* img_ent;
  int img_index = -1;
  int img_count = 0;
  if ((img_dir = opendir(images_dir.c_str())) != NULL) {
    while ((img_ent = readdir(img_dir)) != NULL) {
      std::string f_name = img_ent->d_name;
      img_index++;
      if (!strcmp(f_name.c_str(), ".") || !strcmp(f_name.c_str(), "..")) {
        continue;
      }
      std::string img_file = images_dir + f_name;
      ifstream training_images;
      training_images.open(img_file, ifstream::in);
      float pixel;
      Image_t img =
          vector<vector<float>>(kImageSize, vector<float>(kImageSize));
      for (size_t row = 0; row < 28; row++) {
        for (size_t col = 0; col < 28; col++) {
          if (training_images.read(reinterpret_cast<char*>(&pixel),
                                   sizeof(pixel))) {
            img[row][col] = pixel;
          }
        }
      }
      images_.push_back(img);
      img_count++;
    }
  }

  for (size_t i = 0; i < images_.size(); i++) {
    indices_.push_back(i);
  }
  std::random_shuffle(indices_.begin(), indices_.end());
}

std::vector<Layer> Network::GetLayers() {
  return layers_;
}
void Network::SaveNetwork(std::string& save_file_name) {
  std::ofstream save_file;
  save_file.open(save_file_name, std::ios::out | std::ios::binary);
  for (const Layer& layer : layers_) {
    for (size_t i = 0; i < layer.GetWeights().size(); i++) {
      for (size_t j = 0; j < layer.GetWeights()[i].size(); j++) {
        float weight = layer.GetWeights()[i][j];
        save_file.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
      }
    }
    for (const Neuron& neuron : layer.GetNeurons()) {
      float activation = neuron.GetActivation();
      save_file.write(reinterpret_cast<const char*>(&activation),
                      sizeof(activation));
    }
  }
  save_file.close();
}
void Network::LoadNetwork(std::string& load_file_name) {
  std::ifstream load_file;
  load_file.open(load_file_name, std::ifstream::in | std::ios::binary);
  for (Layer& layer : layers_) {
    for (size_t i = 0; i < layer.GetWeights().size(); i++) {
      for (size_t j = 0; j < layer.GetWeights()[i].size(); j++) {
        float weight;
        load_file.read(reinterpret_cast<char*>(&weight), sizeof(weight));
        layer.SetWeight(i, j, weight);
      }
    }
    for (Neuron& neuron : layer.GetNeurons()) {
      float activation;
      load_file.read(reinterpret_cast<char*>(&activation), sizeof(activation));
      neuron.SetActivation(activation);
    }
  }
  load_file.close();
}
size_t Network::MakePrediction(Image_t img) {
  layers_[0].LoadInputActivations(img);
  layers_[1].ForwardPassHidden(layers_[0]);
  std::vector<float> outputs = layers_[2].ForwardPassOutput(layers_[1]);
  float prediction = -1;
  float max = 0;
  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i] > max) {
      max = outputs[i];
      prediction = i;
    }
  }
  return prediction;
//  for (float value : outputs) {
//    if (value > output) {
//      output = value;
//    }
//  }
//  if (output > positive_threshold_) {
//    return 1;
//  } else {
//    return 0;
//  }
}
void Network::LoadTestingData(std::string& images_dir,
                              std::string& labels_dir) {
  test_images_.clear();
  test_labels_.clear();
  // code for iterating over directory from:
  // https://stackoverflow.com/questions/612097/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-c
  DIR* lbl_dir;
  struct dirent* lbl_ent;
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
      if (training_labels.read(reinterpret_cast<char*>(&label),
                               sizeof(label))) {
//        if ((size_t)label == kPositiveClass) {
//          test_labels_.push_back(1);
//        } else {
//          test_labels_.push_back(0);
//        }
        test_labels_.push_back(size_t(label));
        lbl_count++;
      }
    }
  }

  DIR* img_dir;
  struct dirent* img_ent;
  int img_index = -1;
  int img_count = 0;
  if ((img_dir = opendir(images_dir.c_str())) != NULL) {
    while ((img_ent = readdir(img_dir)) != NULL) {
      std::string f_name = img_ent->d_name;
      img_index++;
      if (!strcmp(f_name.c_str(), ".") || !strcmp(f_name.c_str(), "..")) {
        continue;
      }
      std::string img_file = images_dir + f_name;
      ifstream training_images;
      training_images.open(img_file, ifstream::in);
      float pixel;
      Image_t img =
          vector<vector<float>>(kImageSize, vector<float>(kImageSize));
      for (size_t row = 0; row < 28; row++) {
        for (size_t col = 0; col < 28; col++) {
          if (training_images.read(reinterpret_cast<char*>(&pixel),
                                   sizeof(pixel))) {
            img[row][col] = pixel;
          }
        }
      }
      test_images_.push_back(img);
      img_count++;
    }
  }
}
void Network::ValidateNetwork() {
//  float true_positives = 0;
//  float false_positives = 0;
//  float false_negatives = 0;
//  for (size_t i = 0; i < test_images_.size(); i++) {
//    size_t prediction = MakePrediction(test_images_[i]);
//    if (test_labels_[i] == 1) {
//      if (prediction == 1) {
//        true_positives++;
//      } else {
//        false_negatives++;
//      }
//    } else {
//      if (prediction == 1) {
//        false_positives++;
//      }
//    }
//  }
//  float precision = true_positives / (true_positives + false_positives);
//  float recall = true_positives / (true_positives + false_negatives);
//  std::cout << "Precision: " << precision << ", Recall: " << recall
//            << std::endl;
  float correct = 0;
  float incorrect = 0;
  for (size_t i = 0; i < test_images_.size(); i++) {
    size_t prediction = MakePrediction(test_images_[i]);
    size_t answer = test_labels_[i];
    if (prediction == answer) {
      correct++;
    } else {
      incorrect++;
    }
  }
  float accuracy = correct / (correct + incorrect);
  std::cout << "Accuracy: " << accuracy << std::endl;
}
}  // namespace neural_net