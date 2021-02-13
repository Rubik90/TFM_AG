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
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning) 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import preprocessing, datasets, layers, models

num_features = 64
num_labels = 7
batch_size = 32
epochs = 30
width, height = 48, 48

def load_data():

    df=pd.read_csv('../../datasets/fer.csv')

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

def fit(model,X_train, train_y):

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    cnn_history = model.fit(X_train, train_y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split = 0.4,
              use_multiprocessing = True,
              #validation_steps = 270,
              #validation_data=(X_test, test_y),
              shuffle=True,
              callbacks = [callback])

    return model, cnn_history

def save_loss(cnn_history):

        # Loss plotting
    plt.plot(cnn_history.history['loss'])
    plt.plot(cnn_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('../media/lossResnetFer.png')
    plt.close()

def save_accuracy(cnn_history):

        # Accuracy plotting
    plt.plot(cnn_history.history['accuracy'])
    plt.plot(cnn_history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('../media/accuracyResnetFer.png')
    plt.close()

def save_results(model,X_test, test_y):
#test loss and accuracy
    search_start = time.time()
    loss, accuracy = model.evaluate(X_test, test_y)
    search_end = time.time()
    elapsed_time = search_end - search_start
    print("Elapsed time (s): "+str(elapsed_time))
    print("Test loss: " + str(loss) + "\nTest accuracy: " + str(accuracy))

#save accuracy and loss on file
    test_loss, test_acc = model.evaluate(X_test, test_y)

    print(f"\n Test Loss: {test_loss}, Test Accuracy: {test_acc}")

    f = open("../results/results_resnetFer.txt", "w")
    f.write(f"\n Test Loss: {test_loss}, Test Accuracy: {test_acc}")
    f.close()

#Saving the  model to  use it later on
    mod_json = model.to_json()
    with open("./models/vidModelResnetFer.json", "w") as json_file:
        json_file.write(mod_json)
    model.save_weights("../models/vidModelWeightsResnetFer.h5")

X_train,train_y,X_test,test_y, X_priv, priv_y = load_data()
model=build()
model=compile(model)
model, history = fit(model,X_train, train_y)
save_loss(history)
save_accuracy(history)
save_results(model,X_test, test_y)



