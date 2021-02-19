import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


train_path = "../../datasets/shuffled_balanced/train_frames"
val_path = "../../datasets/shuffled_balanced/val_frames"
test_path = "../../datasets/shuffled_balanced/test_frames"

IMG_HEIGHT = 112
IMG_WIDTH = 112
NUM_CHANNELS = 3
NUM_CLASSES = len(os.listdir(train_path))
batch_size = 8
epochs = 15

def init(train_path, val_path, test_path):
    # Define local directories where datasets are stored
    samples_dir = {
      "train": train_path,
      "val": val_path,
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
      class_names = ["Neutral","Anger","Disgust","Fear","Happiness","Sadness","Surprise"],
      color_mode = "rgb",
      batch_size = 32,
      image_size = (IMG_HEIGHT, IMG_WIDTH),
      shuffle = True,
      validation_split = None,
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

#Compliling the model
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

data_generation = create_data_generator(train_path, val_path, test_path)

model = KerasClassifier(build_fn=c_model, epochs=epochs, batch_size=batch_size)
parameters = {'optimizer':['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']}
clf = GridSearchCV(model, parameters)
clf.fit(X_train, y_train)

print(clf.best_score_, clf.best_params_)
means = clf.cv_results_['mean_test_score']
parameters = clf.cv_results_['params']
for mean, parammeter in zip(means, parameters):
    print(mean, parammeter)