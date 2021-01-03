//
// Created by ash on 11/14/20.
//
#pragma once
#include <stdio.h>

#include <istream>

#include "common_types.h"
#include "layer.h"

namespace neural_net {

class Network {
 public:
  Network(size_t image_size);

  /**
   * Runs the training functionality through every layer
   */
  void Train();

  /**
   * Returns the number of hidden layers in the network
   * @return the number of hidden layers in the network
   */
  size_t GetNumHiddenLayers();

  /**
   * Back propagates the error of the network and calculates deltas
   * @param label the ground truth value of the current image used for training
   */
  void BackPropagation(size_t label);

  /**
   * Calculates loss using cross entropy
   * @param output_activation the final output of the network
   * @param ground_truth label of the current image used for training
   * @return the loss of the network
   */
  float CalculateLoss(std::vector<float> output_activations, size_t ground_truth);

  /**
   * Loads the network's training data
   * @param images_dir directory of images
   * @param labels_dir directory of labels
   */
  void LoadTrainingData(std::string& images_dir, std::string& labels_dir);

  /**
   * Load's the network's testing data
   * @param images_dir directory of images
   * @param labels_dir directory of labels
   */
  void LoadTestingData(std::string& images_dir, std::string& labels_dir);

  /**
   * Saves the weights and activations of the network into a binary file
   * @param save_file_name name of the file that the network will be saved to
   */
  void SaveNetwork(std::string& save_file_name);

  /**
   * Loads weights and activations from a binary file
   * @param load_file_name name of the file to load from
   */
  void LoadNetwork(std::string& load_file_name);

  /**
   * Predicts whether or not the given image is the network's number or not
   * @param img image to be evaluated
   * @return the network's prediction
   */
  size_t MakePrediction(Image_t img);

  /**
   * Runs predictions on every image in the networks' testing data and
   * calculates the network's precision and recall
   */
  void ValidateNetwork();

  std::vector<Layer> GetLayers();
 private:
  const size_t kPositiveClass = 4;
  const size_t kImageSize;
  const size_t kNumClasses = 10;

  size_t num_hidden_layers_ = 1;
  std::vector<Layer> layers_;
  float learning_rate_ = 0.1;
  float positive_threshold_ = 0.48;

  std::vector<Image_t> images_;
  std::vector<size_t> labels_;
  std::vector<size_t> indices_;
  std::vector<Image_t> test_images_;
  std::vector<size_t> test_labels_;
  size_t batch_size = 100;
  size_t hidden_layer_size = 256;
};
} // namespace neural_net