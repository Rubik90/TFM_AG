import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from tensorflow.keras import preprocessing, datasets, layers, models
import os
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

train_path = "/content/shuffled_balanced/train_frames"
val_path = "/content/shuffled_balanced/val_frames"
test_path = "/content/shuffled_balanced/test_frames"

class load_data:
  def __init__(self, train_path, val_path, test_path):
    # Define local directories where datasets are stored
    self.samples_dir = {
      "train": train_path,
      "val": val_path,
      "test": test_path
    }
    self.IMG_HEIGHT = 112
    self.IMG_WIDTH = 112
    self.NUM_CHANNELS = 3
    self.NUM_CLASSES = len(os.listdir(train_path))
  
  def create_data_generator(self):
    data_generator = {}

    for split in self.samples_dir:
      data_generator[split] = preprocessing.image_dataset_from_directory(
        self.samples_dir[split],
        labels = "inferred",
        label_mode = "categorical",
        class_names = ["Neutral","Anger","Disgust","Fear","Happiness","Sadness","Surprise"],
        color_mode = "rgb",
        batch_size = 32,
        image_size = (self.IMG_HEIGHT, self.IMG_WIDTH),
        shuffle = True,
        validation_split = None,
        subset = None,
        interpolation = "gaussian",
        follow_links = False
      )

      # Optimize the dataset using buffered prefetching to avoid blocking I/O
      data_generator[split] = data_generator[split].prefetch(buffer_size = 32)
  
    return data_generator

def c_model(activation):
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

#Compliling the model
    model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

    return model

ld = load_data(train_path, val_path, test_path)
data_generation=ld.create_data_generator()

model = KerasClassifier(build_fn=c_model, epochs=epochs, batch_size=batch_size)
parameters = {'activation':['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']}
clf = GridSearchCV(model, parameters)
clf.fit(data_generation["train"])

print(clf.best_score_, clf.best_params_)
means = clf.cv_results_['mean_test_score']
parameters = clf.cv_results_['params']
for mean, parammeter in zip(means, parameters):
    print(mean, parammeter)