import sys, os
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils
from tensorflow.keras.applications.resnet50 import ResNet50
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50

import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import preprocessing, datasets, layers, models
from tensorflow.keras.applications.resnet50 import ResNet50


train_path = "../dataset/train_frames"
val_path = "../dataset/val_frames"
test_path = "../dataset/test_frames"

IMG_HEIGHT = 112
IMG_WIDTH = 112
NUM_CHANNELS = 3
NUM_CLASSES = len(os.listdir(train_path))
path = "./Con2DAffwild.h5"
batch_size = 8
epochs = 15

def init(train_path, val_path, test_path):
    # Define local directories where datasets are stored
    samples_dir = {
      "train": train_path,
      "validation": val_path,
      "test": test_path
    }
    return samples_dir

def create_data_generator(train_path, val_path, test_path):

  samples_dir = init(train_path, val_path, test_path)
  data_generator = {}

  for split in samples_dir:
    data_generator[split] = preprocessing.image_dataset_from_directory(
      samples_dir[split],
      labels = "inferred",
      label_mode = "categorical",
      class_names = ["0", "1", "2", "3", "4", "5", "6"],
      color_mode = "rgb",
      batch_size = 4,
      image_size = (IMG_HEIGHT, IMG_WIDTH),
      shuffle = True,
      seed = None,
      validation_split = None,
      subset = None,
      interpolation = "gaussian",
      follow_links = False
    )

      # Optimize the dataset using buffered prefetching to avoid blocking I/O
    data_generator[split] = data_generator[split].prefetch(buffer_size = 32)
  
  return data_generator

#1st convolution layer
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH, NUM_CHANNELS)))
model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

#2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

#3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(Flatten())

#fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))
#model.add(LSTM(64,return_sequences=True))

#model = ResNet50(include_top = True, weights = None, input_shape = (X_train.shape[1:]), pooling = "avg", classes = num_labels)
model.summary()

#Compliling the model
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

data_generation = create_data_generator(train_path, val_path, test_path)
#Training the model
cnn_history = model.fit(data_generation["train"],
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          #validation_split = 0.33,
          validation_data=data_generation["validation"],
          shuffle=True)

        # Loss plotting
plt.plot(cnn_history.history['loss'])
plt.plot(cnn_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')
plt.close()

        # Accuracy plotting
plt.plot(cnn_history.history['accuracy'])
plt.plot(cnn_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.png')

#test loss and accuracy
search_start = time.time()
loss, accuracy = model.evaluate(data_generator["test"])
search_end = time.time()
elapsed_time = search_end - search_start
print("Elapsed time (s): "+str(elapsed_time))
print("Test loss: " + str(loss) + "\nTest accuracy: " + str(accuracy))

#Saving the  model to  use it later on
mod_json = model.to_json()
with open("vidModelConv2DFer.json", "w") as json_file:
    json_file.write(mod_json)
model.save_weights("vidModelWeightsConv2DFer.h5")


