'''
Summary:
Loads the saved model and its weights

Returns: Compiled model and graph

'''

from keras.models import load_model
import tensorflow as tf
import os

BASE_DIR = '/Users/NavSha/Documents/tensorflow-projects/dogs-vs-cats/'

def init():
	# load model and its weights from the .h5 file.
	loaded_model = load_model(os.path.join(BASE_DIR,"model/cats_and_dogs_small_1.h5"))
	print("Loaded model from disk")
	graph = tf.get_default_graph()
	return loaded_model,graph
