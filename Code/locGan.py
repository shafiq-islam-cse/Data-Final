from IPython import get_ipython;   
get_ipython().magic('reset -sf')
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
from keras.preprocessing.image import load_img, img_to_array

# All the categories you want your neural network to detect
CATEGORIES = ["r1l1t1", "r2l2t2", "r3l3t3","r1r2,t1t2,l1l2","r1r3,t1t3,l1l3","r2r3,t2t3,l2l3","r1r2r3,t1t2t3,l1l2l3"]

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

Path = "D:/Atik/pythonScripts/Data Final/Partition/Blade Set/CEEMD/Fault localize/"
trainData1 = Path + 'Train'
valData = Path + 'Val'
testData = Path + 'Test17'

x_train1, y_train1 = import_data(trainData1)
x_val , y_val = import_data(valData)
x_test, y_test = import_data(testData)

# Building the classifier
input_shape = (IMG_SIZE, IMG_SIZE, channel)

dir = "D:\\Atik\\pythonScripts\\Data Final\\Partition\\Blade Set\\NEEEMD\\Fault localize\\Fake"

def import_image(size, type):
    images = []
    for i in range(1,1501):
        img = load_img(
            r'{}\{}\1 ({}).png'.format(dir,type,i),
                   grayscale=False, color_mode="rgb")
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        images.append(img)
       
    x = np.vstack(images)
    random.shuffle(x)
    x = x[0:size,:,:,:]
    return x

size = 50
l0 = import_image(size, CATEGORIES[0])
l1 = import_image(size, CATEGORIES[1])
l2 = import_image(size, CATEGORIES[2])
l3 = import_image(size, CATEGORIES[3])
l4 = import_image(size, CATEGORIES[4])
l5 = import_image(size, CATEGORIES[5])
l6 = import_image(size, CATEGORIES[6])
x_train2 = np.concatenate((l0, l1, l2, l3, l4, l5, l6))
y = []
for i in range (0, len(CATEGORIES)):
    a = np.full(size,i)
    y.append(a)
y = np.array(y)
y_train2 = np.ndarray.flatten(y)

x_train = np.concatenate((x_train1, x_train2))
y_train = np.concatenate((y_train1, y_train2))
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_val.shape[0])
print('Number of images in x_test', x_test.shape[0])

# Importing the required Keras modules containing model and layers
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), padding="same", input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3,3),padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3,3),padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, kernel_size=(3,3),padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(8, kernel_size=(3,3),padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(8, kernel_size=(3,3),padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) # Flattening the 2D arrays for fully connected layers

model.add(Dense(512, activation='relu'))

model.add(Dense(len(CATEGORIES),activation='softmax'))

opt = keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=opt, 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Import the early stopping callback
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
# Define a callback to monitor val_acc
early_stopping = EarlyStopping(monitor='val_loss', 
                       patience=15)
modelCheckpoint = ModelCheckpoint('valCNN_model.hdf5', 
                                  save_best_only = True)

# Train your model using the early stopping callback
h_callback = model.fit(x_train, y_train, batch_size = 32,
           epochs = 100, validation_data = (x_val, y_val),
           callbacks = [early_stopping,modelCheckpoint])

# load weights
model.load_weights("valCNN_model.hdf5")
model.compile(optimizer=opt, 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

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