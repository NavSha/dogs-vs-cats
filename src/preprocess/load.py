'''
Summary:
Loads the saved model and its weights

Returns: Compiled model and graph

'''

import numpy as np
import keras.models
from keras.models import model_from_json
from keras import optimizers
import tensorflow as tf
import os

BASE_DIR = '/Users/NavSha/Documents/tensorflow-projects/dogs-vs-cats/'

def init():
	json_file = open(os.path.join(BASE_DIR,"model/cats_and_dogs_small_1.json"))
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(os.path.join(BASE_DIR,"model/cats_and_dogs_small_1.h5"))
	print("loaded model from disk")
	loaded_model.compile(loss = 'binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['accuracy'])
	graph = tf.get_default_graph()

	return loaded_model,graph
