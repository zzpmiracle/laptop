from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow import keras
import os

from matplotlib import pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
y_test = to_categorical(y_test, 10)
y_train = to_categorical(y_train, 10)

def create_model():
    # design model
    model = Sequential()
    model.add(Convolution2D(25, (5, 5), input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(2, 2))
    # model.add(BatchNormalization(axis=-1,epsilon=0.01))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    model.add(Convolution2D(50, (5, 5)))
    model.add(MaxPooling2D(2, 2))
    # model.add(BatchNormalization(axis=-1, epsilon=0.01))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(50))
    # model.add(BatchNormalization(axis=-1, epsilon=0.01))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model

model_path = 'MNIST.hdf5'
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    model = create_model()

adam = Adam(lr=0.001)
# compile model
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
# training model
history = model.fit(x_train, y_train, batch_size=64, epochs=10,verbose=2,validation_split=0.2,shuffle=True)
# test model
# model.save(model_path)
#98.6
print(model.evaluate(x_test, y_test, batch_size=64))
print(history.history)
