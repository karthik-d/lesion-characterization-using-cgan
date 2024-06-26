import matplotlib.pyplot as plt
import numpy as np
import PIL
import os
from pathlib import Path

import tensorflow as tf
# import tensorflow_datasets as tfds
import keras
from keras import layers
from keras.layers import Dense, Flatten, Input
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from .efficientnet.efficientnet.model import EfficientNetB7


BATCH_SIZE = 32
IMG_HEIGHT, IMG_WIDTH = (256, 256)

EPOCHS_REFINE = 20
EPOCHS_TRAIN = 40

MODELS_PATH = os.path.join(
  os.path.dirname(os.path.abspath(__file__)), 
  'models',
  'effnet-b7'
) 

if not os.path.isdir(MODELS_PATH):
  Path(MODELS_PATH).mkdir(
    exist_ok=False,
    parents=True
  )


def experiment_effnetb7(data_path):

	"""**Splitting the data into training and validation**"""

	# data_builder = tfds.folder_dataset.ImageFolder(
	# 	data_path,
	# 	shape=(IMG_HEIGHT, IMG_WIDTH, 3)
	# )

	train_ds = tf.keras.utils.image_dataset_from_directory(
		data_path,
		validation_split=0.2,
		subset="training",
		seed=123,
		image_size=(IMG_HEIGHT, IMG_WIDTH),
		batch_size = BATCH_SIZE 
	)

	val_ds = tf.keras.preprocessing.image_dataset_from_directory(
		data_path,
		validation_split=0.2,
		subset="validation",
		seed=123,
		image_size=(IMG_HEIGHT, IMG_WIDTH),
		batch_size =BATCH_SIZE 
	)


	"""**Training the model**"""

	from tensorflow.keras.models import Sequential

	# img_augmentation = Sequential(
	# 	[
	# 		layers.experimental.preprocessing.RandomRotation(factor=0.15),
	# 		layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
	# 		layers.experimental.preprocessing.RandomFlip(),
	# 		layers.experimental.preprocessing.RandomContrast(factor=0.1),
	# 	],
	# 	name="img_augmentation",
	# )

	# inputs = Input(shape=(256, 256, 3))
	# x = img_augmentation(inputs)  
	model = EfficientNetB7(include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), weights="imagenet")
	# Freeze the pretrained weights
	model.trainable = False

	# take a tensor and compute the average value of all values across the entire matrix for each of the input channels.
	x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.2, name="top_dropout")(x)
	outputs = layers.Dense(8, activation="softmax", name="pred")(x)

	# Compile
	model = keras.Model(model.input, outputs, name="EfficientNet")
	optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
	model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
	# print(model.summary())

	# REFINE
	# hist = model.fit(
	# 	data_builder.as_dataset(
	# 		split='train',
	# 		shuffle_files=True,
	# 		batch_size=BATCH_SIZE
	# 	), 
	# 	epochs=EPOCHS_REFINE, 
	# 	validation_data = data_builder.as_dataset(
	# 		split='validation',
	# 		shuffle_files=False
	# 	)
	# )

	# refine_checkpoint = tf.keras.callbacks.ModelCheckpoint(
	# 	filepath=os.path.join(MODELS_PATH, "{epoch:02d}-{val_loss:.2f}-refine.hdf5"),
	# 	save_weights_only=True,
	# 	save_freq='epoch'
	# )
	# hist = model.fit(train_ds, epochs=1, validation_data=val_ds, steps_per_epoch=1, callbacks=[refine_checkpoint])

	# TRAIN
	model.trainable = True
	optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
	model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

	train_checkpoint = tf.keras.callbacks.ModelCheckpoint(
		filepath=os.path.join(MODELS_PATH, "{epoch:02d}-{val_loss:.2f}-train.hdf5"),
		save_weights_only=True,
		save_freq='epoch'
	)
	hist = model.fit(train_ds, epochs=EPOCHS_TRAIN, validation_data=val_ds, callbacks=[train_checkpoint])


	"""**Validation and Testing**"""

	VAL_DATA_COUNT = len(val_ds)

	model = load_model(os.path.join(MODELS_PATH, "epoch-001.h5"))

	y_val = np.zeros(VAL_DATA_COUNT, dtype=object)
	x_val = np.zeros([VAL_DATA_COUNT, 256, 256, 3])
	i=0
	j=0

	for x,y in val_ds:
		for k in x:
			x_val[j, :,:] = k
			j+=1
		for p in y:
			y_val[i] = p
			i+=1

	y_label = np.empty(VAL_DATA_COUNT, dtype=object)
	for i in range(len(y_val)):
		y_label[i] = np.array(y_val[i])

	classnames = train_ds.class_names
	pred = model.predict(x_val, batch_size=64, verbose=1)

	y_true=[]
	for i in range(len(y_label)):
		y_true.append(classnames[y_label[i]])

	output_class=[]
	for i in range(len(pred)):
		output_class.append(classnames[np.argmax(pred[i])])

	Crct_Counter={'AK':0,'MEL':0,'BCC':0,'NV':0,'BKL':0,'DF':0,'VASC':0,'SCC':0}
	Total_Counter={'AK':0,'MEL':0,'BCC':0,'NV':0,'BKL':0,'DF':0,'VASC':0,'SCC':0}
	for i in range(len(pred)):
		Total_Counter[y_true[i]]+=1
		if(y_true[i]==output_class[i]):
			Crct_Counter[y_true[i]]+=1


	"""**Accuracies for each class**"""

	print("Class_name \t\t Crct_pred \t\t  tot_images \t\t Accuracy")
	print("AK\t\t\t",Crct_Counter['AK'],"\t\t\t ",Total_Counter['AK'],"\t\t\t",round(Crct_Counter['AK']/Total_Counter['AK'],2))
	print("MEL\t\t\t",Crct_Counter['MEL'],"\t\t\t ",Total_Counter['MEL'],"\t\t\t",round(Crct_Counter['MEL']/Total_Counter['MEL'],2))
	print("BCC\t\t\t",Crct_Counter['BCC'],"\t\t\t ",Total_Counter['BCC'],"\t\t\t",round(Crct_Counter['BCC']/Total_Counter['BCC'],2))
	print("NV\t\t\t",Crct_Counter['NV'],"\t\t\t ",Total_Counter['NV'],"\t\t\t",round(Crct_Counter['NV']/Total_Counter['NV'],2))
	print("BKL\t\t\t",Crct_Counter['BKL'],"\t\t\t ",Total_Counter['BKL'],"\t\t\t",round(Crct_Counter['BKL']/Total_Counter['BKL'],2))
	print("DF\t\t\t",Crct_Counter['DF'],"\t\t\t ",Total_Counter['DF'],"\t\t\t",round(Crct_Counter['DF']/Total_Counter['DF'],2))
	print("VASC\t\t\t",Crct_Counter['VASC'],"\t\t\t ",Total_Counter['VASC'],"\t\t\t",round(Crct_Counter['VASC']/Total_Counter['VASC'],2))
	print("SCC\t\t\t",Crct_Counter['SCC'],"\t\t\t ",Total_Counter['SCC'],"\t\t\t",round(Crct_Counter['SCC']/Total_Counter['SCC'],2))


	"""**Displaying the classification report**"""
	target_names = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
	print(classification_report(y_true, output_class, target_names=target_names))


	"""**Displaying the Confusion Matrix**"""
	confusion_matrix(y_true, output_class, labels=target_names)