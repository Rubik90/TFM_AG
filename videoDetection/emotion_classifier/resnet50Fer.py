import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import preprocessing, datasets, layers, models
from tensorflow.keras.applications.resnet50 import ResNet50
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

class model:
  def __init__(self, train_path, val_path, test_path):
    
    self.batch_size = 64
    self.epochs = 15
    self.IMG_HEIGHT = 48
    self.IMG_WIDTH = 48
    self.NUM_CHANNELS = 3
    self.NUM_CLASSES = len(os.listdir(train_path))
    self.path = "./resnet50Fer.h5"
  
  def create_data_generator(self):

        df=pd.read_csv('dataset.csv')

        print(df.info())

        X_train,train_y,X_test,test_y, X_priv, priv_y=[],[],[],[],[],[]

        for index, row in df.iterrows():
            val=row['pixels'].split(" ")
            try:
                if 'Training' in row['Usage']:
                   X_train.append(np.array(val,'float32'))
                   train_y.append(row['emotion'])
                elif 'PublicTest' in row['Usage']:
                   X_test.append(np.array(val,'float32'))
                   test_y.append(row['emotion'])
                elif 'PrivateTest' in row['Usage']:
                   X_priv.append(np.array(val,'float32'))
                   priv_y.append(row['emotion'])
            except:
                print(f"error occured at index :{index} and row:{row}")


        X_train = np.array(X_train,'float32')
        train_y = np.array(train_y,'float32')
        X_test = np.array(X_test,'float32')
        test_y = np.array(test_y,'float32')
        X_priv = np.array(X_priv,'float32')
        priv_y = np.array(priv_y,'float32')

        train_y=np_utils.to_categorical(train_y, num_classes=num_labels)
        test_y=np_utils.to_categorical(test_y, num_classes=num_labels)
        priv_y=np_utils.to_categorical(priv_y, num_classes=num_labels)

#normalizing data between 0 and 1
        X_train -= np.mean(X_train, axis=0)
        X_train /= np.std(X_train, axis=0)

        X_test -= np.mean(X_test, axis=0)
        X_test /= np.std(X_test, axis=0)

        X_priv -= np.mean(X_priv, axis=0)
        X_priv /= np.std(X_priv, axis=0)

        X_train = X_train.reshape(X_train.shape[0], width, height, 1)

        X_test = X_test.reshape(X_test.shape[0], width, height, 1)

        X_priv = X_test.reshape(X_priv.shape[0], width, height, 1)
  
        return X_train,train_y,X_test,test_y, X_priv, priv_y
  
  def build(self, print_summary = False):
    model = ResNet50(include_top = True, weights = None, input_shape = (self.IMG_HEIGHT, self.IMG_WIDTH, self.NUM_CHANNELS), pooling = "avg", classes = self.NUM_CLASSES)

    if print_summary:
      model.summary()

    return model

  def compile(self, model):
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
    
    model.compile(loss  = "categorical_crossentropy",
                  optimizer = optimizer,
                  metrics = ["accuracy"])
    
    return model

  def fit(self, model, X_train,train_y):
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    history = model.fit((X_train,train_y),
                        validation_data = (X_test,test_y),
                        epochs = self.epochs,
                        callbacks = [callback])
    return model, history

  def plot_accuracy(self, history):
    plt.plot(history.history["accuracy"], label = "Train Accuracy")
    plt.plot(history.history["val_accuracy"], label = "Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xscale("linear")
    plt.yscale("linear")
    plt.grid(True)
    plt.ylim([0, 1])
    plt.legend(loc = "lower right")
    plt.show()

    plt.savefig("accuracy.png")
    plt.close()
  
  def plot_loss(self, history):
    plt.plot(history.history["loss"], label = "Train Loss")
    plt.plot(history.history["val_loss"], label = "Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xscale("linear")
    plt.yscale("linear")
    plt.grid(True)
    plt.ylim([0, 1])
    plt.legend(loc = "lower right")
    plt.show()

    # Saves the diagram for further use
    plt.savefig('loss.png')
    plt.close()
  
  def evaluate(self, model, data_generator):
    # Model evaluation
    test_loss, test_acc = model.evaluate(data_generator["test"])

    print(f"\n Test Loss: {test_loss}, Test Accuracy: {test_acc}")

    f = open("Test Evaluation Results.txt", "w")
    f.write(f"\n Test Loss: {test_loss}, Test Accuracy: {test_acc}")
    f.close()

  def save(self, model):
    model.save(self.path)

    print(f"Model saved in {self.path}")
  
  def load(self):
    model = tf.keras.models.load_model(self.path)

    return model

  def predict_generator(self, data_generator):
    prediction = model.predict_generator(generator = data_generator)
    
    print(prediction)

  def run(self):
    X_train,train_y,X_test,test_y, X_priv, priv_y = self.create_data_generator()
    model = self.build(print_summary = True)
    model = self.compile(model)
    model, history = self.fit(model, data_generator)
    self.plot_accuracy(history)
    self.plot_accuracy(history)
    self.evaluate(model, X_priv, priv_y)

    return model, history