import keras
from keras.datasets import mnist
#load mnist dataset
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle

# All the categories you want your neural network to detect
CATEGORIES = ["loss", "rub", "twist"]

# The size of the images that your neural network will use
IMG_SIZE = 64
channel = 3

# Checking or all images in the data folder
def import_data(DATADIR):
    training_data = []
    for category in CATEGORIES :
    	path = os.path.join(DATADIR, category)
    	for img in os.listdir(path):
    		img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_UNCHANGED)
            
    for category in CATEGORIES :
   		path = os.path.join(DATADIR, category)
   		class_num = CATEGORIES.index(category)
   		for img in os.listdir(path):
   			try :
   				img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_UNCHANGED)
   				#new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
   				training_data.append([img_array, class_num])
   			except Exception as e:
   				pass

    random.shuffle(training_data)
        
    X = [] #features
    y = [] #labels
    
    for features, label in training_data:
    	X.append(features)
    	y.append(label)
    
    #X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, channel)
    X = np.array(X)
    
    x_train = np.squeeze(X)
    y_train = np.array(y)
    
    return x_train, y_train

trainData3 = "D:/Atik/pythonScripts/WCNN/Dataset/cnnData/Blade/Train/3"
testData3 = "D:/Atik/pythonScripts/WCNN/Dataset/cnnData/Blade/Test/3"
trainData4 = "D:/Atik/pythonScripts/WCNN/Dataset/cnnData/Blade/Train/4"
testData4 = "D:/Atik/pythonScripts/WCNN/Dataset/cnnData/Blade/Test/4"

x_train3, y_train3 = import_data(trainData3)
x_test3, y_test3 = import_data(testData3)
x_train4, _ = import_data(trainData4)
x_test4, _ = import_data(testData4)

# Building the classifier
input_shape = (64, 64, 3)
# Making sure that the values are float so that we can get decimal points after division
x_train3 = x_train3.astype('float32')
x_test3 = x_test3.astype('float32')
x_train4 = x_train4.astype('float32')
x_test4 = x_test4.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train3 /= 255
x_test3 /= 255
x_train4 /= 255
x_test4 /= 255
print('x_train shape:', x_train3.shape)
print('Number of images in x_train', x_train3.shape[0])
print('Number of images in x_test', x_test3.shape[0])

# Importing the required Keras modules containing model and layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers import Input
from keras.models import Model

inputs1 = Input(shape=(64, 64, 3))
conv1 = Conv2D(64, kernel_size=(3,3))(inputs1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
flat1 = Flatten()(pool1)

inputs2 = Input(shape=(64, 64, 3))
conv2 = Conv2D(64, kernel_size=(3,3))(inputs2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
flat2 = Flatten()(pool2)

merged = concatenate([flat1, flat2])
dense1 = (Dense(128, activation='relu'))(merged)
drop1 = Dropout(0.2)(dense1)
outputs = Dense(4,activation='softmax')(drop1)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
print(model.summary())

model.fit(x=[x_train3, x_train4], y= y_train3, epochs=50, batch_size=16)

loss, acc = model.evaluate([x_test3, x_test4], y_test3, verbose=0)
print('Test Accuracy: %f' % (acc*100))
