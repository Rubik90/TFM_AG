import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import preprocessing, datasets, layers, models
from tensorflow.keras.applications.resnet50 import ResNet50

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
        class_names = ["0", "1", "2", "3", "4", "5", "6"],
        color_mode = "rgb",
        batch_size = 128,
        image_size = (self.IMG_HEIGHT, self.IMG_WIDTH),
        shuffle = True,
        seed = 123,
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
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    
    model.compile(loss  = "categorical_crossentropy",
                  optimizer = optimizer,
                  metrics = ["accuracy"])
    
    return model

  def fit(self, model, data_generator):
    #callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    history = model.fit(data_generator["train"],
                        validation_data = data_generator["val"],
                        epochs = 5)
                        #callbacks = [callback])
    return model, history

  def plot_accuracy(self, history):
    print(history.history)
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
    plt.ylabel("Loss")
    plt.xscale("linear")
    plt.yscale("linear")
    plt.grid(True)
    plt.legend(loc = "lower right")
    plt.show()

    # Saves the diagram for further use
    plt.savefig('loss.png')
    plt.close()
  
  def evaluate(self, model, data_generator):
    # Model evaluation
    test_loss, test_acc = model.evaluate(data_generator["test"])

    print(f"\n Test Loss: {test_loss}, Test Accuracy: {test_acc}")

    f = open("test_evaluation_results.txt", "w")
    f.write(f"\n Test Loss: {test_loss}, Test Accuracy: {test_acc}")
    f.close()

  def save(self, model):
    # Serialize model to JSON
    model_json = model.to_json()

    with open("resnet.json", "w") as json_file:             
        json_file.write(model_json) 

    # Serialize weights to HDF5
    model.save_weights("resnet.h5")

    print("Model saved to disk")
  
  def load(self):
    model = tf.keras.models.load_model(self.path)

    return model

  def predict_generator(self, data_generator):
    prediction = model.predict_generator(generator = data_generator)
    
    print(prediction)

  def run(self):
    data_generator = self.create_data_generator()
    model = self.build(print_summary = True)
    model = self.compile(model)
    model, history = self.fit(model, data_generator)
    self.plot_accuracy(history)
    self.plot_loss(history)
    self.evaluate(model, data_generator)

    return model, history
