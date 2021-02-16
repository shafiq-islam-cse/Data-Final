import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random

# All the categories you want your neural network to detect
CATEGORIES = ["r1l1t1", "r2l2t2", "r3l3t3","r1r2,t1t2,l1l2","r1r3,t1t3,l1l3",\
              "r2r3,t2t3,l2l3","r1r2r3,t1t2t3,l1l2l3"]

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

path = "D:/Atik/pythonScripts/Data Final/Partition/Blade Set/Fault localize"
trainData3 = "%s/Train/3" %path
testData3 = "%s/Test/3" %path
trainData4 = "%s/Train/4" %path
testData4 = "%s/Test/4" %path
valData3 = "%s/Val/3" %path
valData4 = "%s/Val/4" %path

x_train3, y_train3 = import_data(trainData3)
x_test3, y_test3 = import_data(testData3)
x_train4, _ = import_data(trainData4)
x_test4, _ = import_data(testData4)
x_val3 , y_val3 = import_data(valData3)
x_val4 , _ = import_data(valData3)

# Building the classifier
input_shape = (64, 64, 3)
# Making sure that the values are float so that we can get decimal points after division
x_train3 = x_train3.astype('float32')
x_test3 = x_test3.astype('float32')
x_train4 = x_train4.astype('float32')
x_test4 = x_test4.astype('float32')
x_val3 = x_val3.astype('float32')
x_val4 = x_val4.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
x_train3 /= 255
x_test3 /= 255
x_train4 /= 255
x_test4 /= 255
x_val3 /= 255
x_val4 /= 255
print('x_train shape:', x_train3.shape)
print('Number of images in x_train', x_train3.shape[0])
print('Number of images in x_test', x_test3.shape[0])
print('Number of images in x_val', x_val3.shape[0])

# Importing the required Keras modules containing model and layers
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers import Input
from keras.models import Model

inputsX1 = Input(shape=(64, 64, 3))
convX1 = Conv2D(64,kernel_size=(3,3),padding="same")(inputsX1)
poolX1 = MaxPooling2D(pool_size=(2, 2))(convX1)

convX2 = Conv2D(32, kernel_size=(3,3),padding="same")(poolX1)
poolX2 = MaxPooling2D(pool_size=(2, 2))(convX2)

convX3 = Conv2D(32, kernel_size=(3,3),padding="same")(poolX2)
poolX3 = MaxPooling2D(pool_size=(2, 2))(convX3)

convX4 = Conv2D(16, kernel_size=(3,3),padding="same")(poolX3)
poolX4 = MaxPooling2D(pool_size=(2, 2))(convX4)

convX5 = Conv2D(8, kernel_size=(3,3),padding="same")(poolX4)
poolX5 = MaxPooling2D(pool_size=(2, 2))(convX5)

convX6 = Conv2D(8, kernel_size=(3,3),padding="same")(poolX5)
poolX6 = MaxPooling2D(pool_size=(2, 2))(convX6)

flat1 = Flatten()(poolX6)

inputsY1 = Input(shape=(64, 64, 3))
convY1 = Conv2D(64,kernel_size=(3,3),padding="same")(inputsY1)
poolY1 = MaxPooling2D(pool_size=(2, 2))(convY1)

convY2 = Conv2D(32, kernel_size=(3,3),padding="same")(poolY1)
poolY2 = MaxPooling2D(pool_size=(2, 2))(convY2)

convY3 = Conv2D(32, kernel_size=(3,3),padding="same")(poolY2)
poolY3 = MaxPooling2D(pool_size=(2, 2))(convY3)

convY4 = Conv2D(16, kernel_size=(3,3),padding="same")(poolY3)
poolY4 = MaxPooling2D(pool_size=(2, 2))(convY4)

convY5 = Conv2D(8, kernel_size=(3,3),padding="same")(poolY4)
poolY5 = MaxPooling2D(pool_size=(2, 2))(convY5)

convY6 = Conv2D(8, kernel_size=(3,3),padding="same")(poolY5)
poolY6 = MaxPooling2D(pool_size=(2, 2))(convY6)

flat2 = Flatten()(poolY6)

merged = concatenate([flat1, flat2])
dense1 = (Dense(512, activation='relu'))(merged)
drop1 = Dropout(0.3)(dense1)
outputs = Dense(len(CATEGORIES),activation='softmax')(drop1)
model = Model(inputs=[inputsX1, inputsY1], outputs=outputs)
        
opt = keras.optimizers.Adam(learning_rate=1e-03)
model.compile(optimizer=opt, 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
print(model.summary())

# Import the early stopping callback
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# Define a callback to monitor val_acc
early_stopping = EarlyStopping(monitor='val_loss', 
                       patience=15)
modelCheckpoint = ModelCheckpoint('valCNN_model.hdf5', 
                                  save_best_only = True)

h_callback = model.fit(x=[x_train3, x_train4], y= y_train3, epochs=100, batch_size=32,
          validation_data = ([x_val3,x_val4], y_val3), 
          callbacks = [early_stopping,modelCheckpoint])

loss, acc = model.evaluate([x_test3, x_test4], y_test3, verbose=0)
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
  
# Plot train vs test loss during training
plot_loss(h_callback.history['loss'], (h_callback.history['val_loss']))

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
  
# Plot train vs test accuracy during training
plot_accuracy(h_callback.history['accuracy'], h_callback.history['val_accuracy'])

from sklearn.metrics import confusion_matrix
pred = model.predict([x_test3, x_test4])
pred = pred.argmax(axis = 1)
conf = confusion_matrix(y_test3, pred)
print(conf)
print('acc : ', np.trace(conf)/np.sum(conf)*100)