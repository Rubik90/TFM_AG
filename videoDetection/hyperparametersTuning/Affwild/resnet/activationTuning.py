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
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf
train_path = "../../datasets/shuffled_balanced/train_frames"
val_path = "../../datasets/shuffled_balanced/val_frames"
test_path = "../../datasets/shuffled_balanced/test_frames"

IMG_HEIGHT = 112
IMG_WIDTH = 112
NUM_CHANNELS = 3
NUM_CLASSES = len(os.listdir(train_path))

samples_dir = {
      "train": train_path,
      "val": val_path,
      "test": test_path
    }

  
def create_data_generator():
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
        subset = None,
        interpolation = "gaussian",
        follow_links = False
      )

      # Optimize the dataset using buffered prefetching to avoid blocking I/O
      data_generator[split] = data_generator[split].prefetch(buffer_size = 32)
  
    return data_generator

def build():

    model = ResNet50(include_top = True, weights = None, input_shape = (X_train.shape[1:]), pooling = "avg", classes = num_labels)
    model.summary()

    return model


def compile(model):

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    
    model.compile(loss  = "categorical_crossentropy",
                      optimizer = optimizer,
                      metrics = ["accuracy"])

    return model

def c_model():
	data_generator = create_data_generator()
    model=build()
    model=compile(model)
    return model

model = KerasClassifier(build_fn=c_model, epochs=epochs, batch_size=batch_size)
parameters = {'activation':['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']}
clf = GridSearchCV(model, parameters)
clf.fit(X_train, y_train)

print(clf.best_score_, clf.best_params_)
means = clf.cv_results_['mean_test_score']
parameters = clf.cv_results_['params']
for mean, parammeter in zip(means, parameters):
    print(mean, parammeter)