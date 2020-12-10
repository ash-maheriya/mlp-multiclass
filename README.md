##Overview
In this project, I have implemented a Multi-Layer Perceptron neural network. The network is capable of
binary classification. It can be trained to determine whether a given image (a 2-dimensional array of floats)
either does or doesn't belong to a certain group. For example, I have trained the network to determine whether
the given image contains a written number 4 or not. Additionally, I created a visualization of the neural network, which
draws out the nodes of each layer and the weights connecting them, with varied color weights to distinguish their values.

##Requirements
I implemented the network from scratch. The approach is entirely object oriented and should not rely on any
external libraries that are not a part of C++ already. However, this repository does not include the data that
I used, as there are tens of thousands of images. I would recommend visiting the MNIST database (http://yann.lecun.com/exdb/mnist/),
where you can choose the dataset(s) that you want to work with. Keep in mind that you will want to select data that
can be distinctly categorized into two groups, even if the grouping is as simple as: a) image of a dog b) not an image of a dog.
I used a Python script to convert files from the MNIST database into binary files.

##Functionality
The network is capable of training, given a dataset with images and corresponding labels. The network will periodically
save its training progress (its weights and activation values) to a binary file. The network is also capable of reading
back these values from the binary file to pick up where it left off. train_model_main.cc contains a sample run of what the
network does. I included validation functionality, which allows users to check the precision and recall of the network they are building.

##Important Notes
The network is hardcoded to have one input layer, one hidden layer, and one output layer. Additionally, the network is only
designed to take in images of the same size. For example, I trained the network with 28x28 images.

##References
I had to do a lot of research to understand how to implement a neural network. I would recommend checking out Andrew Ng's
course on machine learning on Coursera (you can view the course at your own pace for free), and http://neuralnetworksanddeeplearning.com/
as good sources.