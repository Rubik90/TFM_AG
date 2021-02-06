import sys, os
import pandas as pd
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from loguru import logger
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

LOG_DIR = f"{int(time.time())}"
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

from tuner_comparison import (
    INPUT_SHAPE,
    NUM_CLASSES,
    N_EPOCH_SEARCH,
)

df=pd.read_csv('dataset.csv')

# print(df.info())
# print(df["Usage"].value_counts())

# print(df.head())
X_train,train_y,X_test,test_y=[],[],[],[]

for index, row in df.iterrows():
      val=row['pixels'].split(" ")
      try:
          if 'Training' in row['Usage']:
            X_train.append(np.array(val,'float32'))
            train_y.append(row['emotion'])
          elif 'PublicTest' in row['Usage']:
             X_test.append(np.array(val,'float32'))
             test_y.append(row['emotion'])
      except:
        print(f"error occured at index :{index} and row:{row}")


num_features = 64
num_labels = 7
batch_size = 64
epochs = 30
width, height = 48, 48


X_train = np.array(X_train,'float32')
train_y = np.array(train_y,'float32')
X_test = np.array(X_test,'float32')
test_y = np.array(test_y,'float32')

train_y=np_utils.to_categorical(train_y, num_classes=num_labels)
test_y=np_utils.to_categorical(test_y, num_classes=num_labels)

#cannot produce
#normalizing data between oand 1
X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

# print(f"shape:{X_train.shape}")
##designing the cnn
#1st convolution layer

def build_model():
    
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])))
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

    model.add(Dense(num_labels, activation='softmax'))
#model.add(LSTM(64,return_sequences=True))

    model.summary()

#Compliling the model
    model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

    logger.info("Start training")
    search_start = time.time()
    return model
#Training the model
    """cnn_history = model.fit(X_train, train_y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_test, test_y),
              shuffle=True)
    search_end = time.time()
    elapsed_time = search_end - search_start
    logger.info(f"Elapsed time (s): {elapsed_time}")

    loss, accuracy = model.evaluate(X_test, test_y)
    logger.info(f"loss: {loss}, accuracy: {accuracy}")
        return model

plt.plot(cnn_history.history["accuracy"], label = "Train Accuracy")
plt.plot(cnn_history.history["val_accuracy"], label = "Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xscale("linear")
plt.yscale("linear")
plt.grid(True)
plt.ylim([0, 1])
plt.legend(loc = "lower right")
plt.savefig("accuracy.png")
plt.close()
        # Loss plotting
plt.plot(cnn_history.history['loss'])
plt.plot(cnn_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')
plt.close()

#Saving the  model to  use it later on
mod_json = model.to_json()
with open("mod.json", "w") as json_file:
    json_file.write(mod_json)
model.save_weights("mod.h5")
model.save("mod_to_plot.h5")"""

tuner = RandomSearch(
	  build_model,
	  objective = "val_accuracy",
	  max_trials =1,
	  executions_per_trial=1,
	  directory = LOG_DIR)

tuner.search(x= X_train,
		y=y_train,
		epochs=1,
		batch_size=64,
		validation_data=(X_test,test_y))