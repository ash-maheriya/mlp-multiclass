## Overview
In this project, I have implemented a Multi-Layer Perceptron neural network. The network is capable of
multi-class classification. For example, I have trained the network to determine what digit (from 0 to 9) is contained
in the image. This is a sigmoid based network with a softmax output layer. I have trained it with the MNIST digits dataset with an accuracy of 95.9%. 
Additionally, I created a Cinder-based visualization of the neural network, which draws out the nodes of each layer and the weights connecting them, 
with varied color weights to distinguish their values.

## Requirements
I implemented the network from scratch. The approach is entirely object oriented and should not rely on any
external libraries that are not a part of C++ already. However, this repository does not include the data that
I used, as there are tens of thousands of images. I would recommend visiting the MNIST database (http://yann.lecun.com/exdb/mnist/),
where you can choose the dataset(s) that you want to work with. Keep in mind that you will want to select data that
can be distinctly categorized into classes. I used a Python script to convert files from the MNIST database into binary files. I would also recommend running the code from
an IDE like CLion. A compressed file containing the dataset I used for training can be found here: https://drive.google.com/drive/folders/1-7JMfSP7h9O27JhR6VnU7b5iPgfxmz5Z?usp=sharing

## Installation
The project executables can be run without an IDE. After cloning the repository, create a build directory and then run cmake from within the
main project folder. Then, run make from within the build directory you created. This should create executables for training (main), visualizing,
and testing the network that you can run from the command line.

## Functionality
The network is trained using a set of images and a corresponding set of labels. The network will periodically
save its training progress (its weights and activation values) to a binary file. The network is also capable of reading
back these values from the binary file to pick up where it left off. train_model_main.cc contains a sample run of what the
network does. I included validation functionality, which allows users to check the accuracy of the network being built.

## Important Notes
The network is hardcoded to have one input layer, one hidden layer, and one output layer (this can be changed within the
network.cc file. Additionally, the network is only designed to take in images of the same size. 
For example, I trained the network with 28x28 images (standard MNIST size).

## References
I had to do a lot of research to understand how to implement a neural network. I would recommend checking out Andrew Ng's
course on machine learning on Coursera (you can view the course at your own pace for free), and http://neuralnetworksanddeeplearning.com/
as good sources.