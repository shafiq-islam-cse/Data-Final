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

trainData = "D:/Atik/pythonScripts/WCNN/Dataset/cnnData/Blade/Train/3"
valData = "D:/Atik/pythonScripts/WCNN/Dataset/cnnData/Blade/Val/3"
testData = "D:/Atik/pythonScripts/WCNN/Dataset/cnnData/Blade/Test/3"

x_train, y_train = import_data(trainData)
x_val , y_val = import_data(valData)
x_test, y_test = import_data(testData)

# Building the classifier
input_shape = (64, 64, 3)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_val.shape[0])
print('Number of images in x_test', x_test.shape[0])

# Importing the required Keras modules containing model and layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4,activation='softmax'))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Import the early stopping callback
from tensorflow.keras.callbacks import EarlyStopping
# Define a callback to monitor val_acc
early_stopping = EarlyStopping(monitor='val_loss', 
                       patience=15)

# Train your model using the early stopping callback
h_callback = model.fit(x_train, y_train, 
           epochs = 100, validation_data = (x_val, y_val),
           callbacks = [early_stopping])

loss, acc= model.evaluate(x_test, y_test)
print('Test Accuracy: %f' % (acc*100))

def plot_loss(loss,val_loss):
  plt.figure()
  plt.plot(loss)
  plt.plot(val_loss)
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Validation'], loc='upper right')
  plt.show()
  
def plot_accuracy(acc,val_acc):
  # Plot training & validation accuracy values
  plt.figure()
  plt.plot(acc)
  plt.plot(val_acc)
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Validation'], loc='upper left')
  plt.show()
  
# Plot train vs test loss during training
plot_loss(h_callback.history['loss'], h_callback.history['val_loss'])

# Plot train vs test accuracy during training
plot_accuracy(h_callback.history['accuracy'], h_callback.history['val_accuracy'])

from sklearn.metrics import confusion_matrix
pred = model.predict(x_test)
pred = pred.argmax(axis = 1)
conf = confusion_matrix(y_test, pred)
print(conf)
print('acc : ', np.trace(conf)/np.sum(conf)*100)