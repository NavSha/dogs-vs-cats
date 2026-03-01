'''
Summary:
	Trains a dogs vs cats classifier using MobileNetV2
	transfer learning (two-phase: frozen base → fine-tune).

Returns:
	Saves model (.h5 + .json), architecture diagram, and
	training curves to the model/ directory.
'''

import os
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import plot_model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

IMG_WIDTH, IMG_HEIGHT = 150, 150
BATCH_SIZE = 32

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MODEL_DIR = os.path.join(PROJECT_DIR, 'model')

base_dir = os.environ.get('CATS_DOGS_DATA_DIR', os.path.join(PROJECT_DIR, 'data'))
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

MODEL_PATH = os.path.join(MODEL_DIR, 'cats_and_dogs_mobilenet.h5')


def create_transfer_model():
	'''Build MobileNetV2 with custom classification head.'''
	base_model = MobileNetV2(weights='imagenet', include_top=False,
							 input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
	base_model.trainable = False

	model = models.Sequential([
		base_model,
		layers.GlobalAveragePooling2D(),
		layers.Dense(256, activation='relu'),
		layers.Dropout(0.5),
		layers.Dense(1, activation='sigmoid')
	])
	return model


def get_data_generators():
	'''Set up augmented training and plain validation generators.'''
	train_datagen = ImageDataGenerator(
		rescale=1./255,
		rotation_range=40,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest')
	val_datagen = ImageDataGenerator(rescale=1./255)

	train_gen = train_datagen.flow_from_directory(
		train_dir, target_size=(IMG_WIDTH, IMG_HEIGHT),
		batch_size=BATCH_SIZE, class_mode='binary')
	val_gen = val_datagen.flow_from_directory(
		validation_dir, target_size=(IMG_WIDTH, IMG_HEIGHT),
		batch_size=BATCH_SIZE, class_mode='binary')

	return train_gen, val_gen


def training():
	train_gen, val_gen = get_data_generators()
	steps_per_epoch = train_gen.samples // BATCH_SIZE
	validation_steps = val_gen.samples // BATCH_SIZE

	model = create_transfer_model()

	# --- Save architecture diagram ---
	plot_model(model, to_file=os.path.join(MODEL_DIR, 'architecture.png'),
			   show_shapes=True, show_layer_names=True)
	print("Architecture diagram saved to model/architecture.png")

	# --- Phase 1: Feature extraction (frozen base) ---
	print("\n=== Phase 1: Feature extraction (frozen base) ===")
	model.compile(
		loss='binary_crossentropy',
		optimizer=optimizers.RMSprop(learning_rate=1e-4),
		metrics=['accuracy'])

	log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	callbacks_p1 = [
		ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss',
						verbose=1, save_best_only=True),
		EarlyStopping(monitor='val_loss', patience=5, verbose=1,
					  restore_best_weights=True),
		ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3),
		TensorBoard(log_dir=log_dir, histogram_freq=0),
	]

	history_p1 = model.fit(
		train_gen, steps_per_epoch=steps_per_epoch, epochs=30,
		validation_data=val_gen, validation_steps=validation_steps,
		callbacks=callbacks_p1)

	# Save model after Phase 1
	model.save(MODEL_PATH)
	model_json = model.to_json()
	with open(os.path.join(MODEL_DIR, 'cats_and_dogs_mobilenet.json'), "w") as f:
		f.write(model_json)
	print(f"\nPhase 1 model saved to {MODEL_PATH}")

	# --- Phase 2: Fine-tuning last 30 layers ---
	history_p2 = None
	try:
		print("\n=== Phase 2: Fine-tuning last 30 layers ===")
		base_model = model.layers[0]
		base_model.trainable = True
		for layer in base_model.layers[:-30]:
			layer.trainable = False

		model.compile(
			loss='binary_crossentropy',
			optimizer=optimizers.RMSprop(learning_rate=1e-5),
			metrics=['accuracy'])

		log_dir_ft = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_finetune"
		callbacks_p2 = [
			ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss',
							verbose=1, save_best_only=True),
			EarlyStopping(monitor='val_loss', patience=5, verbose=1,
						  restore_best_weights=True),
			ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3),
			TensorBoard(log_dir=log_dir_ft, histogram_freq=0),
		]

		history_p2 = model.fit(
			train_gen, steps_per_epoch=steps_per_epoch, epochs=20,
			validation_data=val_gen, validation_steps=validation_steps,
			callbacks=callbacks_p2)

		model.save(MODEL_PATH)
		print(f"\nFine-tuned model saved to {MODEL_PATH}")
	except Exception as e:
		print(f"\nPhase 2 failed ({e}), using Phase 1 model.")

	return history_p1, history_p2


def plot_loss_and_accuracy(history_p1, history_p2=None):
	'''Plot training curves for both phases (or just Phase 1) and save to file.'''
	acc = history_p1.history['accuracy']
	val_acc = history_p1.history['val_accuracy']
	loss = history_p1.history['loss']
	val_loss = history_p1.history['val_loss']
	phase1_epochs = len(acc)

	if history_p2 is not None:
		acc += history_p2.history['accuracy']
		val_acc += history_p2.history['val_accuracy']
		loss += history_p2.history['loss']
		val_loss += history_p2.history['val_loss']

	epochs = range(1, len(acc) + 1)

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

	ax1.plot(epochs, acc, 'bo-', label='Training acc')
	ax1.plot(epochs, val_acc, 'b-', label='Validation acc')
	if history_p2 is not None:
		ax1.axvline(x=phase1_epochs, color='gray', linestyle='--', label='Fine-tune start')
	ax1.set_title('Training and Validation Accuracy')
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Accuracy')
	ax1.legend()

	ax2.plot(epochs, loss, 'ro-', label='Training loss')
	ax2.plot(epochs, val_loss, 'r-', label='Validation loss')
	if history_p2 is not None:
		ax2.axvline(x=phase1_epochs, color='gray', linestyle='--', label='Fine-tune start')
	ax2.set_title('Training and Validation Loss')
	ax2.set_xlabel('Epoch')
	ax2.set_ylabel('Loss')
	ax2.legend()

	plt.tight_layout()
	save_path = os.path.join(MODEL_DIR, 'training_curves.png')
	plt.savefig(save_path, dpi=150)
	print(f"Training curves saved to {save_path}")


if __name__ == "__main__":
	history_p1, history_p2 = training()
	plot_loss_and_accuracy(history_p1, history_p2)
