import os,shutil
original_dataset_dir = '/Users/NavSha/Documents/tensorflow-projects/Cat_Dog_data'
original_dataset_dir_cat = '/Users/NavSha/Documents/tensorflow-projects/Cat_Dog_data/Cat/'
original_dataset_dir_dog = '/Users/NavSha/Documents/tensorflow-projects/Cat_Dog_data/Dog'
#set base directory
base_dir = '/Users/NavSha/Documents/tensorflow-projects/cats_and_dogs_small'
os.mkdir(base_dir)

#create training, validation and test directories
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir,'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir,'test')
os.mkdir(test_dir)

# create dogs and cats directories inside training, validation and test directories
train_cats_dir = os.path.join(train_dir,'cats')
os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir,'dogs')
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir,'cats')
os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir,'dogs')
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir,'cats')
os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir,'dogs')
os.mkdir(test_dogs_dir)

# copy 1000 cats images to the cats training directory
fnames = ['{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_cat, fname)
    dst = os.path.join(train_cats_dir,fname)
    shutil.copyfile(src,dst)
# copy 500 cats images to cats validation directory
fnames = ['{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_cat, fname)
    dst = os.path.join(validation_cats_dir,fname)
    shutil.copyfile(src,dst)
#copy 500 cats images to the cats testing directory
fnames = ['{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_cat, fname)
    dst = os.path.join(test_cats_dir,fname)
    shutil.copyfile(src,dst)

# copy 1000 dogs images to the cats training directory
fnames = ['{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_dog, fname)
    dst = os.path.join(train_dogs_dir,fname)
    shutil.copyfile(src,dst)
#copy 500 dogs images to the dogs validation directoy
fnames = ['{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_dog, fname)
    dst = os.path.join(validation_dogs_dir,fname)
    shutil.copyfile(src,dst)
#copy 500 dogs images to the testing directory
fnames = ['{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_dog, fname)
    dst = os.path.join(test_dogs_dir,fname)
    shutil.copyfile(src,dst)

#define the model
from keras import layers
from keras import models

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
model.summary()

#compile the model
from keras import optimizers
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])

# data preprocessing using ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=(150,150),batch_size=20,class_mode='binary')

history = model.fit_generator(train_generator, steps_per_epoch=100,epochs=10,validation_data = validation_generator,validation_steps=50)
model.save('cats_and_dogs_small_1.h5')

#let's plot the training and validation losses and accuracies
import matplotlib.pyplot as plt

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
