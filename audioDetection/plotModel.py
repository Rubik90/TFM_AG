import keras
from keras.utils import plot_model

from config import MODEL_DIR_PATH

restored_keras_model = keras.models.load_model(MODEL_DIR_PATH + 'audioModel.h5')

        cnn_history = model.fit(x_traincnn, y_train,
                               batch_size=16, epochs=50,
                               validation_data=(x_testcnn, y_test))

        model.summary()
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
plt.show()
predictions = model.predict_classes(x_testcnn)
new_y_test = y_test.astype(int)
matrix = confusion_matrix(new_y_test, predictions)

print(classification_report(new_y_test, predictions))
print(matrix)

plot_model(restored_keras_model, to_file='media/model.png')