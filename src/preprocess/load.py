'''
Summary:
Loads the saved model and its weights

Returns: Compiled model

'''

from tensorflow.keras.models import load_model
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def init():
	# load model and its weights from the .h5 file.
	loaded_model = load_model(os.path.join(PROJECT_DIR,"model/cats_and_dogs_mobilenet.h5"))
	print("Loaded model from disk")
	return loaded_model
