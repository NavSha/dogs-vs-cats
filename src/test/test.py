'''
Summary:
	Tests whether an image is a dog or a cat
    by comparing it with true labels.

Returns:
	It returns a predicted.csv file containing all the
    predictions and wrong-predictions.csv file containing
    all the predictions that were wrong.
'''

from keras.models import model_from_json
import os
import sys
from keras.preprocessing.image import ImageDataGenerator
sys.path.append(os.path.abspath("../preprocess"))
from load import *
global model, graph
model, graph = init()

#set up base and test directories
base_dir = '/Users/NavSha/Documents/tensorflow-projects/cats_and_dogs_small'
test_dir = os.path.join(base_dir,'test')

def test_classifier():

    # loads the model and prints its accuracy over the test dataset
	test_datagen = ImageDataGenerator(rescale=1./255)
	test_generator = test_datagen.flow_from_directory(test_dir,target_size=(150,150),batch_size=20,class_mode='binary')
	test_loss, test_acc = model.evaluate_generator(test_generator,steps = 50)
	print('test acc: ',test_acc)

if __name__ == "__main__":
    test_classifier()
