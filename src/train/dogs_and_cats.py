'''
Summary:
	Loads the dataset and trains the classifier
    model on dogs and cats images

Returns:
	Returns JSON as well as H5 file for the model
    which can be used by server for classification
    and deployment
'''

import os
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import datetime
os.environ['KMP_DUPLICATE_LIB_OK']='True'

IMG_WIDTH, IMG_HEIGHT = 150, 150

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

#set up base, training, validation and test directories
base_dir = os.environ.get('CATS_DOGS_DATA_DIR', os.path.join(PROJECT_DIR, 'data'))
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir,'validation')
test_dir = os.path.join(base_dir,'test')

def create_model():
# create a model with 3 conv layers and 3 maxpooling layers
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(150,150,3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation = 'relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation = 'relu'))
    model.add(layers.Dense(1,activation = 'sigmoid'))
    return model

def training():
    # data preprocessing using ImageDataGenerator
	train_datagen = ImageDataGenerator(
		rescale=1./255,
		rotation_range=40,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest')
	test_datagen = ImageDataGenerator(rescale=1./255)
	train_generator = train_datagen.flow_from_directory(train_dir,target_size=[IMG_WIDTH,IMG_HEIGHT],batch_size=20,class_mode='binary')
	validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=[IMG_WIDTH,IMG_HEIGHT],batch_size=20,class_mode='binary')

	model = create_model()
    #compile the model
	model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-4),metrics=['accuracy'])
	checkpointer = ModelCheckpoint(filepath=os.path.join(PROJECT_DIR, 'model/cats_and_dogs_small_1.h5'), monitor = 'val_loss', verbose = 1, save_best_only=True, mode='auto')
	stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto',restore_best_weights=True)

	learning_rate_update = ReduceLROnPlateau(monitor = 'val_loss',factor = 0.1, patience = 3)
	log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0)

	history = model.fit(train_generator, steps_per_epoch=100,epochs=3,validation_data = validation_generator,validation_steps=50, callbacks=[checkpointer, stop, learning_rate_update, tensorboard_callback])
	model.save(os.path.join(PROJECT_DIR, 'model/cats_and_dogs_small_1.h5'))

	model_json = model.to_json()
	with open(os.path.join(PROJECT_DIR, 'model/cats_and_dogs_small_1.json'), "w") as json_file:
		json_file.write(model_json)
	return history

def plot_loss_and_accuracy():
#let's plot the training and validation losses and accuracies

	history = training()
	acc = history.history['accuracy']
	loss = history.history['loss']
	val_acc = history.history['val_accuracy']
	val_loss = history.history['val_loss']

	epochs = range(1,len(acc)+1)

	plt.plot(epochs, acc, 'bo', label = 'Training acc')
	plt.plot(epochs,val_acc,'b',label = 'Validation acc')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.figure()

	plt.plot(epochs, loss, 'ro', label = 'Training loss')
	plt.plot(epochs,val_loss,'r',label = 'Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.show()

if __name__=="__main__":
	plot_loss_and_accuracy()
