'''
Summary:
	Tests whether an image is a dog or a cat
    by comparing it with true labels.

Returns:
	It prints the accuracy of the model on the test dataset
'''

import os
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "../preprocess"))
from load import *
global model
model = init()

#set up base and test directories
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
base_dir = os.environ.get('CATS_DOGS_DATA_DIR', os.path.join(PROJECT_DIR, 'data'))
test_dir = os.path.join(base_dir,'test')

def test_classifier():
	# loads the model and prints its accuracy over the test dataset
	test_datagen = ImageDataGenerator(rescale=1./255)
	test_generator = test_datagen.flow_from_directory(test_dir,target_size=(150,150),batch_size=20,class_mode='binary')
	test_loss, test_acc = model.evaluate(test_generator,steps = 50)
	print('test acc: ',test_acc)

if __name__ == "__main__":
    test_classifier()
