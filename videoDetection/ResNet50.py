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


num_features = 64
num_labels = 7
batch_size = 64
epochs = 15
width, height = 48, 48


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

import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import preprocessing, datasets, layers, models
from tensorflow.keras.applications.resnet50 import ResNet50
print(X_train.shape[1:])
model = ResNet50(include_top = True, weights = None, input_shape = (X_train.shape[1:]), pooling = "avg", classes = num_labels)
model.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    
model.compile(loss  = "categorical_crossentropy",
                  optimizer = optimizer,
                  metrics = ["accuracy"])

cnn_history = model.fit(X_train, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          #validation_split = 0.33,
          validation_data=(X_test, test_y),
          shuffle=True,
          callbacks = [callback])

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

#validation loss and accuracy
search_start = time.time()
loss, accuracy = model.evaluate(X_test, test_y)
search_end = time.time()
elapsed_time = search_end - search_start
print("Elapsed time (s): "+str(elapsed_time))
print("Validation loss: " + str(loss) + "\nValidation accuracy: " + str(accuracy))

#test loss and accuracy
search_start = time.time()
loss, accuracy = model.evaluate(X_priv, priv_y)
search_end = time.time()
elapsed_time = search_end - search_start
print("Elapsed time (s): "+str(elapsed_time))
print("Test loss: " + str(loss) + "\nTest accuracy: " + str(accuracy))

#Saving the  model to  use it later on
mod_json = model.to_json()
with open("vidModel.json", "w") as json_file:
    json_file.write(mod_json)
model.save_weights("vidModelWeights.h5")




