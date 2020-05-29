import os
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Convolution2D, Dense, Flatten, Activation, MaxPooling2D

from tensorflow.keras.models import load_model

data_path = 'D:\python\dataset\Dog_Cat'
train_image_path = os.path.join(data_path, 'train')
test_image_path = os.path.join(data_path, 'test')
img_width, img_height = 32, 32
image_dim = (img_width, img_height, 3)
epochs = 10
batch_size = 32

train_data_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(directory=train_image_path,
                                                                                target_size=(img_width, img_height),
                                                                                class_mode='binary',
                                                                                batch_size=batch_size)
test_data_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(directory=test_image_path,
                                                                               target_size=(img_width, img_height),
                                                                               class_mode='binary',
                                                                               batch_size=batch_size)

model_path = 'dog_cat_model.hdf5'


def create_model():
    model = Sequential()
    model.add(Convolution2D(25, (5, 5), input_shape=image_dim))
    model.add(MaxPooling2D(2, 2))
    model.add(Activation('relu'))
    model.add(Convolution2D(50, (5, 5)))
    model.add(MaxPooling2D(2, 2))
    model.add(Activation('relu'))
    model.add(Flatten())

    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model
#
#
# if os.path.exists(model_path):
#     model = load_model(model_path)
# else:
#     model = create_model()
# model.compile(loss='binary_crossentropy',
#               optimizer='Adadelta',
#               metrics=['accuracy'])
#
# model_history = model.fit(train_data_generator,
#                           epochs=epochs,
#                           verbose=2, )
# model.save(model_path)
# print(model_history.history)
# score = model.evaluate_generator(test_data_generator)
# print(score[-1])

from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation



def resnet_layer(inputs,num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x
def resnet_v2(input_shape, depth, num_classes=10):
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation='relu',
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='sigmoid',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

filepath='dog_cat_ResNet.hdf5'

if os.path.exists(filepath):
    model = load_model(filepath)
else:
    model = resnet_v2(depth=20,
                      num_classes=1,
                     input_shape=(image_dim))
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png')
model.compile(loss='binary_crossentropy',
                           optimizer='Adam',
                           metrics=['accuracy'])

modet_history = model.fit_generator(train_data_generator,
                                    epochs=epochs,
                                    )
model.save(filepath)
score = model.evaluate_generator(test_data_generator)
print(score)

