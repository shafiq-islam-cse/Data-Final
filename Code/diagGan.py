from IPython import get_ipython;   
get_ipython().magic('reset -sf')
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random

# All the categories you want your neural network to detect
CATEGORIES = ["Loss","Rub","Twist"]

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
    
    x_train = np.squeeze(X).astype('float32')
    x_train /= 255
    y_train = np.array(y).astype('float32')
    
    return x_train, y_train

Path = "D:/Atik/pythonScripts/Data Final/Partition/Blade Set/CEEMD/Fault diagnose/"
trainData = Path + 'Train'
valData = Path + 'Val'
testData = Path + 'Test'

x_train1, y_train1 = import_data(trainData)
x_val , y_val = import_data(valData)
x_test, y_test = import_data(testData)

dir = "D:/Atik/pythonScripts/Data Final/Gan/Blade/Fake/"
CATEGORIES = ["l1","l1l2","l1l2l3","l1l3","l2","l2l3","l3"]
x_trainA, _ = import_data(dir + 'loss')
y_trainA = np.zeros(len(x_trainA))
CATEGORIES = ["r1","r1r2","r1r2r3","r1r3","r2","r2r3","r3"]
x_trainB , _ = import_data(dir + 'rub')
y_trainB = np.ones(len(x_trainB))
CATEGORIES = ["t1","t1t2","t1t2t3","t1t3","t2","t2t3","t3"]
x_trainC, _ = import_data(dir + 'twist')
y_trainC = np.full(len(x_trainC),2)

size = 1000

x_train = np.concatenate((x_train1, x_trainA[0:size,:,:,:],
                          x_trainB[0:size,:,:,:], x_trainC[0:size,:,:,:]))
y_train = np.concatenate((y_train1, y_trainA[0:size], 
                          y_trainB[0:size], y_trainC[0:size]))

# Building the classifier
input_shape = (64, 64, 3)
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_val.shape[0])
print('Number of images in x_test', x_test.shape[0])

# Importing the required Keras modules containing model and layers
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), padding="same", activation='relu', input_shape=x_train[0].shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3,3),activation='relu',padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3,3),activation='relu',padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(96, kernel_size=(3,3),activation='relu',padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) # Flattening the 2D arrays for fully connected layers

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(len(CATEGORIES),activation='softmax'))

opt = keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=opt, 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Import the early stopping callback
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
# Define a callback to monitor val_acc
early_stopping = EarlyStopping(monitor='val_loss', 
                       patience=15)
modelCheckpoint = ModelCheckpoint('diagCNN_model.hdf5', 
                                  save_best_only = True)
    
# Train your model using the early stopping callback
h_callback = model.fit(x_train, y_train, batch_size = 16,
           epochs = 100, validation_data = (x_val, y_val),
           callbacks = [early_stopping,modelCheckpoint])

# load weights
model.load_weights("diagCNN_model.hdf5")

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
  plt.legend(['Train', 'Validation'], loc='lower right')
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