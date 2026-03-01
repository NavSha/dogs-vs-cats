## Binary Classification using CNNs: Dogs vs Cats

A project to test new state-of-the-art models in computer vision and deep learning. I'm doing this project primarily for my own relearning purpose. 

**Dogs vs Cats** was a very popular problem in image classification a few years ago and many teams from the world competed on Kaggle to get the best possible accuracy on it. This problem turned out to be too hard for valina neural networks and required deep convolutional neural networks.  

In this repo, I've added a model that consists of 4 conv layers and 4 maxpooling layers. It takes as input images in (150,150,3) format and passes it through all the layers successively. The outout from the final maxpooling layer is then flattened and passed as input to a dense network which uses a sigmoid function for the binary classification. 

## Prerequisites

* **Python** 3.9+
* **TensorFlow** 2.14+
* **OpenCV** 4.8+
* **Flask** 3.0+

## How to run

* Create a virtual environment and install dependencies

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

* Download the dataset and place it in `data/` (should contain `train/`, `validation/`, `test/` subdirectories). Link to the training dataset is provided in `src/train/README.md`.

* Start the server

```
python src/server/server.py
```

* Open http://localhost:5000 in your browser to use the drag-and-drop UI, or use curl:

```
curl -X POST -F 'image=@src/test/1510.jpg' http://localhost:5000/api/class_pred
```

* To evaluate the model on the test dataset:

```
python src/test/test.py
```

# Contributing

Feel free to clone it and make changes to it. Pull requests are welcome.

# Author

Navneet Sharma

# Acknowledgements

Deep Learning for Python Book, Keras Documentation etc.

# Loss and Validation Curves for the Model

![alt text](https://github.com/NavSha/dogs-vs-cats/blob/master/src/pictures/loss_and_accuracy.png)

# Sample Output 

Below are some of the images from the test dataset along with predictions from the trained model

------
