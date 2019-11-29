'''
Summary:
	Loads the dataset and trains the classifier
    model on dogs and cats images

Returns:
	Returns JSON as well as H5 file for the model
    which can be used by server for classification
    and deployment
'''

import os,shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import backend as K
os.environ['KMP_DUPLICATE_LIB_OK']='True'

IMG_WIDTH, IMG_HEIGHT = 150, 150

#set up base, training, validation and test directories
base_dir = '/Users/NavSha/Documents/tensorflow-projects/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir,'validation')
test_dir = os.path.join(base_dir,'test')

def img_channel_type():
    # Pre-preprocessing of image
    if K.image_data_format() == 'channels_first':
        input_shape = (3, IMG_WIDTH, IMG_HEIGHT)
    else:
        input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)

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
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir,target_size=[IMG_WIDTH,IMG_HEIGHT],batch_size=20,class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=[IMG_WIDTH,IMG_HEIGHT],batch_size=20,class_mode='binary')

    model = create_model()
    #compile the model
    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])

    history = model.fit_generator(train_generator, steps_per_epoch=100,epochs=10,validation_data = validation_generator,validation_steps=50)
    model.save('cats_and_dogs_small_1.h5')

    model_json = model.to_json()
    with open('model_ugc.json', "w") as json_file:
        json_file.write(model_json)

def plot_loss_and_accuracy():
#let's plot the training and validation losses and accuracies
    acc = history.history['acc']
    loss = history.history['loss']
    val_acc = history.history['val_acc']
    val_loss = history.history['val_loss']

    epochs = range(1,len(acc)+1)

    plt.plot(epochs, acc, 'bo', label = 'Training acc')
    plt.plot(epochs,val_acc,'bo',label = 'Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'ro', label = 'Training loss')
    plt.plot(epochs,val_acc,'ro',label = 'Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

if __name__=="__main__":
    img_channel_type()
    training()
