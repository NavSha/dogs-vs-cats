# Binary Classification using CNNs: Dogs vs Cats

A project to test new state-of-the-art models in computer vision and deep learning. I'm doing this project primarily for my own relearning purpose. 

Dogs vs cats was a very popular probelm in image classification a few years ago and many teams from the world competed on Kaggle to get the best possible accuracy on it. This problem turned out to be too hard for valina neural networks and required deep convolutional neural networks.  

In this repo, I've added a model that consists of 4 conv layers and 4 maxpooling layers. It takes as input images in (150,150,3) format and passes it through all the layers successively. The outout from the final maxpooling layer is then flattened and passed as input to a 2-layer dense network which uses a sigmoid function for the binary classification. 

# Prerequisites

Python 
Numpy
TensorFlow
Keras
Miniconda

# Contributing

Feel free to clone it and make changes to it. Pull requests are welcome.

# Author

Navneet Sharma

# Acknowledgements

Deep Learning for Python Book

# Loss and Validation Curves for the Model

------

# Sample Output 

Below are some of the images from the test dataset along with predictions from the trained model

------
