'''
Summary:
	Evaluates the trained model on the test dataset
    and prints loss and accuracy.
'''

import os
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

sys.path.append(os.path.join(SCRIPT_DIR, "../preprocess"))
from load import init

model = init()

base_dir = os.environ.get('CATS_DOGS_DATA_DIR', os.path.join(PROJECT_DIR, 'data'))
test_dir = os.path.join(base_dir, 'test')

def evaluate():
	test_datagen = ImageDataGenerator(rescale=1./255)
	test_generator = test_datagen.flow_from_directory(test_dir, target_size=(150,150), batch_size=20, class_mode='binary')
	test_loss, test_acc = model.evaluate(test_generator, steps=50)
	print(f'test loss: {test_loss:.4f}')
	print(f'test acc:  {test_acc:.4f}')

if __name__ == "__main__":
    evaluate()
