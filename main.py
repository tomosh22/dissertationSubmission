import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GaussianNoise, Cropping2D, Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, \
	GlobalAveragePooling2D, RandomFlip, RandomRotation
from sklearn import metrics
from sklearn.model_selection import train_test_split
# import tensorflow_addons as tfa
# from tensorflow_addons import layers as tfaLayers
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util
import pandas as pd
from tensorflow.keras import models
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import csv
import math
import logging


def addnistTrain():
	x_train = np.load("train_x.npy")
	x_train = np.moveaxis(x_train, 1, -1)
	y_train = np.load("train_y.npy")
	y_train = keras.utils.to_categorical(y_train, 20)
	return tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

def addnistTest():
	x_valid = np.load("valid_x.npy")
	x_valid = np.moveaxis(x_valid, 1, -1)
	y_valid = np.load("valid_y.npy")

	x_test = np.load("test_x.npy")
	x_test = np.moveaxis(x_test, 1, -1)
	y_test = np.load("test_y.npy")

	y_valid = keras.utils.to_categorical(y_valid, 20)
	y_test = keras.utils.to_categorical(y_test, 20)

	x_test = np.concatenate((x_test, x_valid))
	y_test = np.concatenate((y_test, y_valid))
	return tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

def addnistModel():
	return Sequential(
			[
				Input(shape=(28, 28, 3)),
				Conv2D(32, 3, strides=(1, 1), padding="valid"),
				MaxPooling2D(),
				Conv2D(32, 3, strides=(1, 1), padding="valid"),
				MaxPooling2D(),
				Conv2D(64, 3, strides=(2, 2), padding="valid"),
				MaxPooling2D(),
				Flatten(),
			]
		)

def flowersTrain():
	train_datagen = ImageDataGenerator(
		# rescale=1. / 255,
		# shear_range=0.2,
		# zoom_range=0.2,
		horizontal_flip=True,
		vertical_flip=True,
		# preprocessing_function=keras.applications.resnet50.preprocess_input,
		# height_shift_range=0.3,
		# width_shift_range=0.3,
		# rotation_range=30
	)
	train_generator = train_datagen.flow_from_directory(
		"flowersLD3fps/New folder/light",
		target_size=(17, 30),
		# batch_size=batch_size,
		class_mode='categorical')
	return train_generator

def flowersTest():
	test_datagen = ImageDataGenerator()
	test_generator = test_datagen.flow_from_directory(
		"flowersLD3fps/New folder/dark",
		target_size=(17, 30),
		class_mode="categorical")
	return test_generator

def flowersModel():
	return Sequential(
		[
			Input(shape=(17,30, 3)),
			Conv2D(4, 3, strides=(1, 1), padding="valid", kernel_regularizer='l1_l2', bias_regularizer='l1_l2',
				   activity_regularizer='l1_l2'),
			MaxPooling2D(),
			Conv2D(8, 2, strides=(2, 2), padding="valid", kernel_regularizer='l1_l2', bias_regularizer='l1_l2',
				   activity_regularizer='l1_l2'),
			MaxPooling2D(),
			# Dropout(0.2),
			Flatten(),
		]
	)

def goodvbadTrain():
	train_datagen = ImageDataGenerator(
		validation_split=0.5,
		# rescale=1. / 255,
		# shear_range=0.2,
		# zoom_range=0.2,
		horizontal_flip=True,
		# vertical_flip=True,
		# height_shift_range=0.3,
		# width_shift_range=0.3,
		# rotation_range=30
	)
	trainFlow = train_datagen.flow_from_directory("goodguysbadguys/train", subset="training", target_size=(50, 62),
												  class_mode="categorical")
	return trainFlow

def goodvbadTest():
	test_datagen = ImageDataGenerator(validation_split=0.5,
									  # rescale=1./255
									  )
	testFlow = test_datagen.flow_from_directory("goodguysbadguys/test", subset="validation", target_size=(50, 62),
												class_mode="categorical")
	return testFlow

def goodvbadModel():
	return Sequential(
		[
			Input(shape=(50,62, 3)),
			Conv2D(16, 8, strides=(1, 1), padding="valid"),
			MaxPooling2D(),
			Dropout(0.2),
			Conv2D(32, 4, strides=(2, 2), padding="valid"),
			MaxPooling2D(),
			Dropout(0.2),
			Conv2D(48, 2, strides=(2, 2), padding="valid"),
			MaxPooling2D(),
			Dropout(0.2),
			Flatten()
		]
	)

def riceTrain():
	train_datagen = ImageDataGenerator(
		validation_split=0.5,
		# rescale=1. / 255,
		# shear_range=0.2,
		# zoom_range=0.2,
		# horizontal_flip=True,
		# vertical_flip=True,
		# height_shift_range=0.3,
		# width_shift_range=0.3,
		# rotation_range=30
	)
	trainFlow = train_datagen.flow_from_directory("Rice_Image_Dataset", subset="training", target_size=(125, 125),
												  class_mode="categorical")
	return trainFlow

def riceTest():
	test_datagen = ImageDataGenerator(validation_split=0.5,
									  # rescale=1./255
									  )
	testFlow = test_datagen.flow_from_directory("Rice_Image_Dataset", subset="validation", target_size=(125, 125),
												class_mode="categorical")
	return testFlow

def riceModel():
	return Sequential(
		[
			Input(shape=(125,125, 3)),
			Conv2D(3, 4, strides=(2, 2), padding="valid"),
			MaxPooling2D(),
			Dropout(0.2),
			Conv2D(4, 3, strides=(3, 3), padding="valid"),
			MaxPooling2D(),
			Dropout(0.2),
			Conv2D(5, 2, strides=(3, 3), padding="valid"),
			MaxPooling2D(),
			Dropout(0.2),
			Flatten()
		]
	)

def mnistCombined():
	(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
	x_train = x_train.astype("float32") / 255
	x_test = x_test.astype("float32") / 255
	y_train = keras.utils.to_categorical(y_train)
	y_test = keras.utils.to_categorical(y_test)
	return (tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32),tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32))

def mnistModel():
	return Sequential(
		[
			Input(shape=(28,28,1)),
			Conv2D(3, 5, strides=(2, 2), padding="valid"),
			MaxPooling2D(),
			Conv2D(5, 3, strides=(2, 2), padding="valid"),
			MaxPooling2D(),
			Flatten(),
		]
	)

if __name__ == "__main__":
	logging.getLogger("tensorflow").disabled = True
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	datasets = {
		"addnist": (addnistTrain(), addnistTest()),
		"flowers": (flowersTrain(),flowersTest()),
		"goodvbad": (goodvbadTrain(),goodvbadTest()),
		"rice": (riceTrain(),riceTest()),
		"mnist": mnistCombined()
	}
	convModels = {
		"addnist": lambda:addnistModel(),
		"flowers": lambda:flowersModel(),
		"goodvbad": lambda:goodvbadModel(),
		"rice": lambda:riceModel(),
		"mnist": lambda:mnistModel()
	}
	inputWidths = {"addnist": 64, "flowers": 24, "goodvbad": 48, "rice": 5, "mnist": 5}
	numClasses = {"addnist": 20, "flowers": 3, "goodvbad": 2, "rice": 5, "mnist": 10}
	widthMultipliers = [1.1,1.25,1.5,2,3,5,10,20]
	depths = range(1,11)
	numEpochs = {"addnist":10,"flowers":20,"goodvbad":10,"rice":10,"mnist":5}
	finalDropouts = {"addnist":0.2,"flowers":None,"goodvbad":0.4,"rice":0.4,"mnist":0.2}
	for name in datasets.keys():
		trainSet = datasets[name][0]
		testSet = datasets[name][1]
		for widthMultiplier in widthMultipliers:
			for depth in depths:
				fileOut = open(f"moreResults/{name}/{str(round(inputWidths[name] * widthMultiplier))}_{str(depth)}.csv", "w")
				fieldNames = ["trainAcc", "testAcc"]
				writer = csv.DictWriter(fileOut, fieldnames=fieldNames)
				writer.writeheader()
				for run in range(10):
					model = convModels[name]()
					for __ in range(depth):
						model.add(Dense(round(inputWidths[name] * widthMultiplier),activation="relu"))
					if finalDropouts[name]:
						model.add(Dropout(finalDropouts[name]))
					model.add(Dense(numClasses[name],activation="softmax"))
					model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
								  metrics=["accuracy"])
					print(f"Training {name} model with depth {depth} and width multiplier {widthMultiplier}. Run {run+1}")
					history = model.fit(trainSet,epochs=numEpochs[name],verbose=0)
					results = model.evaluate(testSet,verbose=0)
					print(f"Train accuracy: {history.history['accuracy'][-1]} Test accuracy: {results[1]}")
					writer.writerow({"trainAcc":history.history["accuracy"][-1],"testAcc":results[1]})
				fileOut.close()
