import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Activation
from keras.models import Sequential
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import LSTM
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import time

from config import SAVE_DIR_PATH
from config import MODEL_DIR_PATH


class audioModel:

    def NN(X, y) -> None:

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        x_traincnn = np.expand_dims(X_train, axis=2)
        x_testcnn = np.expand_dims(X_test, axis=2)

        print(x_traincnn.shape, x_testcnn.shape)

        model = Sequential()
        model.add(Conv1D(64, 5, padding='same',
                         input_shape=(40, 1)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(8))
        model.add(Activation('softmax'))

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])

        cnn_history = model.fit(x_traincnn, y_train,
                               batch_size=128, epochs=180,
                               validation_data= None,
                               validation_split = 0.4)

        model.summary()
        # Loss plotting
        plt.plot(cnn_history.history['loss'])
        plt.plot(cnn_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('./media/loss.png')
        plt.close()

        # Accuracy plotting
        plt.plot(cnn_history.history['accuracy'])
        plt.plot(cnn_history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('./media/accuracy.png')
        plt.show()
        predictions = model.predict_classes(x_testcnn)
        new_y_test = y_test.astype(int)
        matrix = confusion_matrix(new_y_test, predictions)

        print(classification_report(new_y_test, predictions))
        print(matrix)

        #test loss and accuracy
        search_start = time.time()
        loss, accuracy = model.evaluate(x_testcnn, y_test)
        search_end = time.time()
        elapsed_time = search_end - search_start
        print("Elapsed time (s): "+str(elapsed_time))
        print("Test loss: " + str(loss) + "\nTest accuracy: " + str(accuracy))
        model_name = 'audioModel.h5'
        print(f"\n Test Loss: {loss}, Test Accuracy: {accuracy}")

        f = open("./results/results.txt", "w")
        f.write(f"\n Test Loss: {loss}, Test Accuracy: {accuracy}")
        f.close()

        # Save model and weights
        if not os.path.isdir(MODEL_DIR_PATH):
            os.makedirs(MODEL_DIR_PATH)
        model_path = os.path.join(MODEL_DIR_PATH, model_name)
        model.save(model_path)
        print('Saved trained model at %s ' % model_path)


if __name__ == '__main__':
    print('Training started')
    X = joblib.load(SAVE_DIR_PATH + 'X.joblib')
    y = joblib.load(SAVE_DIR_PATH + 'y.joblib')
    audioModel.NN(X=X, y=y)

