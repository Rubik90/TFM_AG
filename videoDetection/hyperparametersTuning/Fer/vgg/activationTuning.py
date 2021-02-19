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
from tensorflow.keras.applications.vgg16 import VGG16

np.random.seed(0)

df=pd.read_csv('../datasets/fer.csv')

X_train,y_train,X_test,test_y=[],[],[],[]

for index, row in df.iterrows():
    val=row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
           X_train.append(np.array(val,'float32'))
           y_train.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
           X_test.append(np.array(val,'float32'))
           test_y.append(row['emotion'])
    except:
        print(f"error occured at index :{index} and row:{row}")


num_features = 64
num_labels = 7
batch_size = 64
num_epochs = 3
width, height = 48, 48


X_train = np.array(X_train,'float32')
y_train = np.array(y_train,'float32')
X_test = np.array(X_test,'float32')
test_y = np.array(test_y,'float32')

y_train=np_utils.to_categorical(y_train, num_classes=num_labels)
test_y=np_utils.to_categorical(test_y, num_classes=num_labels)

#normalizing data between 0 and 1
X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

def build():

    model = VGG16(include_top = True, weights = None, input_shape = (X_train.shape[1:]), pooling = "avg", classes = num_labels)
    model.summary()

    return model

def compile(model):

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    
    model.compile(loss  = "categorical_crossentropy",
                      optimizer = optimizer,
                      metrics = ["accuracy"])

    return model

def c_model(activation):
    X_train,train_y,X_test,test_y, X_priv, priv_y = load_data()
    model=build()
    model=compile(model)

model = KerasClassifier(build_fn=c_model, epochs=epochs, batch_size=batch_size)
parameters = {'activation':['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']}
clf = GridSearchCV(model, parameters)
clf.fit(X_train, y_train)

print(clf.best_score_, clf.best_params_)
means = clf.cv_results_['mean_test_score']
parameters = clf.cv_results_['params']
for mean, parammeter in zip(means, parameters):
    print(mean, parammeter)