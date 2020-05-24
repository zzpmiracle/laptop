from ResNet import resnet_v2
import os

from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.applications import resnet50
from keras.utils import plot_model
from matplotlib import pyplot as plt
import math

from keras.preprocessing.image import ImageDataGenerator

from numpy.random import seed

train_image_path = 'D:\\Event&NoEvent\\train'
val_image_path = 'D:\\Event&NoEvent\\validation'
test_image_path = 'D:\\Event&NoEvent\\test'

nb_train_samples = 4000
nb_val_samples = 500
nb_test_samples = 500

img_width, img_height = 32,32
image_dim = (img_width,img_height, 3)

batch_size = 32
epochs = 60

train_data_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(directory=train_image_path,
                                                          target_size=(img_width,img_height),
                                                          class_mode='binary',
                                                          batch_size=batch_size)
val_data_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(directory=val_image_path,
                            target_size=(img_width,img_height),
                            class_mode='binary',
                            batch_size=batch_size)
test_data_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(directory=test_image_path,
                            target_size=(img_width,img_height),
                            class_mode='binary',
                            batch_size=batch_size)

filepath='weights.best.hdf5'

if os.path.exists(filepath):
    model = load_model(filepath)
else:
    model = resnet_v2(depth=20,
                      num_classes=1,
                     input_shape=image_dim)

model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

check_point = ModelCheckpoint(filepath=filepath,
                              monitor='val_acc',
                              verbose=1,
                              save_best_only='False',
                              mode='max')

history = model.fit_generator(train_data_generator,
                    steps_per_epoch=math.ceil(nb_train_samples/batch_size),
                    epochs=epochs,
                    callbacks=[check_point],
                    verbose=1,
                            validation_data=val_data_generator,
                              validation_steps=math.ceil(nb_val_samples/batch_size))
print(history.history)
model.save(filepath)


# # 绘制训练 & 验证的准确率值
# plt.plot(history.history['acc'])
# # plt.plot(history.history['loss'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# # 绘制训练 & 验证的损失值
# plt.plot(history.history['loss'])
# # plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()


test_data_gen = ImageDataGenerator(rescale=1./255)

test_data_generator = test_data_gen.flow_from_directory(directory=test_image_path,
                            target_size=(img_width,img_height),
                            class_mode='binary',
                            batch_size=batch_size)
score = model.evaluate_generator(test_data_generator,steps=math.ceil(nb_test_samples/batch_size))

print(score[-1])