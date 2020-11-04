
#Project Description
For my final project, I will be creating a simple multilayer perceptron (MLP) neural network. The neural network
will be used to classify written numbers, similar to the Naive Bayes MP. The bulk of the work of the project would be creating
 the functionality for performing the forward pass and backpropagation with one or two hidden layers and fully training the 
 network. So far, I do not think I will be using an external library to do a significant portion of the project. 
 For the Cinder aspect, I will create a visualization of the network, showing the layers and the weights 
 (inspired by https://playground.tensorflow.org/ which I've used before for learning neural networks). 
 
#Background Knowledge
I worked with a MobileNet V2 convolutional neural network in a robotics project previously. The project was in Python, and a
majority of the network functionality was handled by the TensorFlow library. I find machine learning very interesting, and
I want to implement the forward pass and backpropagation backend of a neural network myself in C++. I am choosing a MLP for this project because it is
simpler than a convolutional network and as such can be implemented in the given time frame. I find machine learning very
interesting and plan on working with it more in the future. As such, I want to be familiarize myself with the low-level
implementation of a network.

#Timeline Breakdown  
As a general timeline breakdown for this project, I believe that the first two weeks will be spent actually creating the neural network, and
the last week will be spent setting up the Cinder visualization. I plan on using the first week to implement the calculations
necessary for training the model. This will include elements such as the activation function for each neuron, the mapping of weights between pixels of training images and neurons and between two neurons,
etc. As of now, I plan on implementing it with only one hidden layer, but as a stretch goal, I may create a neural
network that can be trained with a variable, user-input defined number of hidden layers. The second week will be used to
actually train the model. This will involve taking grayscale images from the MNIST database. Depending on how training goes,
I will make tweaks to the functionality of the network. I will use the third week to implement the Cinder part of the project.
The main part of this will be creating a visualization of the neural network that has been trained. As you can see in the
example graph shown on the website linked in the first paragraph, the graph will display all of the features, hidden layers,
and neurons, and more importantly, the weights that are connecting them. The values of the weights are the most significant
result of this neural network, so my Cinder visualization will aim to display these weights in the most interesting way possible.
As a stretch goal, I may create a visualization that shows the evolution of the network from the beginning of training
to the end of it, showing how the weights change over time. I will also use the sketchpad app from the Naive Bayes MP
so that I can show the classification capabilities of the network in real time.

#Stretch Goals
I have listed a couple stretch goals already, but I believe that this project will fit the 3 week work period precisely.
Given how I have planned it out, I believe I will be able to get it done in time. If I see that I will need more than 3
weeks, I will work on the project during Thanksgiving to pick up the slack. If I find that I have extra time, I will try
to implement the stretch goals I have mentioned.