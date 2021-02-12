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
import tensorflow as tf

df=pd.read_csv('../fer.csv')

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
epochs = 30
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

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

X_priv = X_test.reshape(X_priv.shape[0], 48, 48, 1)

##designing the cnn
#1st convolution layer
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

model.add(Dense(num_labels, activation='sigmoid'))
#model.add(LSTM(64,return_sequences=True))

model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

#Compliling the model
model.compile(loss=categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])

#Training the model
cnn_history = model.fit(X_train, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split = 0.4,
          #validation_data=(X_test, test_y),
          shuffle=True)

        # Loss plotting
plt.plot(cnn_history.history['loss'])
plt.plot(cnn_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('lossConv2Fer.png')
plt.close()

        # Accuracy plotting
plt.plot(cnn_history.history['accuracy'])
plt.plot(cnn_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracyConv2Fer.png')

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
with open("vidModelConv2Fer.json", "w") as json_file:
    json_file.write(mod_json)
model.save_weights("vidModelWeightsConv2Fer.h5")


