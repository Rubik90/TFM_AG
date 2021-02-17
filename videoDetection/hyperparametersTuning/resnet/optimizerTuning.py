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
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

class model:
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
  
  def build(self, print_summary = False):
    model = ResNet50(include_top = True, weights = None, input_shape = (self.IMG_HEIGHT, self.IMG_WIDTH, self.NUM_CHANNELS), pooling = "avg", classes = self.NUM_CLASSES)

    if print_summary:
      model.summary()

    return model

  def compile(self, model):
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    
    model.compile(loss  = "categorical_crossentropy",
                  optimizer = optimizer,
                  metrics = ["accuracy"])
    
    return model

def c_model():
  def run():
    resnet = model(train_path, val_path, test_path)
    data_generator = resnet.create_data_generator()
    model = resnet.build(print_summary = True)
    model = resnet.compile(model)

from sklearn.model_selection import GridSearchCV

def c_model():
    resnet = model(train_path, val_path, test_path)
    data_generator = resnet.create_data_generator()
    model = resnet.build(print_summary = True)
    model = resnet.compile(model)
model = KerasClassifier(build_fn=c_model, epochs=50, batch_size=32)
parameters = {'optimizer':['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']}
clf = GridSearchCV(model, parameters)
clf.fit(X_train, y_train)

print(clf.best_score_, clf.best_params_)
means = clf.cv_results_['mean_test_score']
parameters = clf.cv_results_['params']
for mean, parammeter in zip(means, parameters):
    print(mean, parammeter)