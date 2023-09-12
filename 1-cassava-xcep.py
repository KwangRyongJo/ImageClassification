# https://github.com/hokmund/xception-transfer-learning/blob/master/data-preparation.ipynb

from os import listdir, rename
from os.path import isfile, join

source_dir = 'D:/2021/17-01-2021/cassavadata/data/cbsd-cmd-mix'
train_dir = 'D:/2021/17-01-2021/cassavadata/data/train'
validation_dir = 'D:/2021/17-01-2021/cassavadata/data/validation'
test_dir = 'D:/2021/17-01-2021/cassavadata/data/test'

cbsds_dir = 'cbsds'
cmds_dir = 'cmds'

images = [f for f in listdir(source_dir) if isfile(join(source_dir, f))]

cbsds = [i for i in images if i.startswith('train-cbsd')]
cbsds[0]

cmds = [i for i in images if i.startswith('train-cmd')]
cmds[0]

for cbsd in cbsds[:200]:
    rename(join(source_dir, cbsd), join(train_dir, cbsds_dir, cbsd))

for cbsd in cbsds[200:300]:
    rename(join(source_dir, cbsd), join(validation_dir, cbsds_dir, cbsd))

for cbsd in cbsds[300:600]:
    rename(join(source_dir, cbsd), join(test_dir, cbsds_dir, cbsd))


for cmd in cmds[:200]:
    rename(join(source_dir, cmd), join(train_dir, cmds_dir, cmd))

for cmd in cmds[200:300]:
    rename(join(source_dir, cmd), join(validation_dir, cmds_dir, cmd))

for cmd in cmds[300:600]:
    rename(join(source_dir, cmd), join(test_dir, cmds_dir, cmd))


######################################################### 
# https://github.com/hokmund/xception-transfer-learning/blob/master/transfer-learning.ipynb

from tensorflow.keras.models import Model
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import xception
from tensorflow.keras.layers import Input
import numpy as np
import tensorflow as tf

# from keras.backend.tensorflow_backend import set_session

# # We want our experiment to be reproducible
# np.random.seed(42)
# tf.set_random_seed(42)

# This setting allows Tensorflow to allocate GPU memory in runtime rather than at the session initialization.
# config = tf.ConfigProto()
# config = tf.compat.v1.ConfigProto
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))

train_data_dir = 'D:/2021/17-01-2021/cassavadata/data/train'
test_data_dir = 'D:/2021/17-01-2021/cassavadata/data/test'
validation_data_dir = 'D:/2021/17-01-2021/cassavadata/data/validation'

img_height = 299
img_width = 299

# You may need to decrease batch size depending on your GPU capacity
batch_size = 5
epochs = 5

train_first_class = 200
train_second_class = 200

val_first_class = 100
val_second_class = 100

nb_train_samples = train_first_class + train_second_class
nb_validation_samples = val_first_class + val_second_class


input_tensor = Input(shape=(img_height, img_width, 3))

base_model = xception.Xception(weights='imagenet',
                               include_top=False,
                               input_shape=(img_width, img_height, 3),
                               pooling='avg')

data_generator = image.ImageDataGenerator(rescale=1. / 255)

train_generator = data_generator.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

bottleneck_features_train = base_model.predict(
    train_generator,
    nb_train_samples // batch_size)

print('Prediction of the training set finished.')

np.save(open('bottleneck_features_train.npy', 'wb'),
        bottleneck_features_train)

validation_generator = data_generator.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

bottleneck_features_validation = base_model.predict(
    validation_generator,
    nb_validation_samples // batch_size)

print('Prediction of the validation set finished.')

np.save(open('bottleneck_features_validation.npy', 'wb'),
        bottleneck_features_validation)
train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
train_labels = np.array(([0] * train_first_class) + ([1] * train_second_class))

validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
validation_labels = np.array(
    [0] * (val_first_class) + [1] * (val_second_class))


model = Sequential()

model.add(Dense(256, activation='relu',
                input_shape=base_model.output_shape[1:]))
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=SGD(lr=0.005),
              loss='binary_crossentropy',
              metrics=['accuracy'])


checkpointer = ModelCheckpoint(
    filepath='top-weights.h5', verbose=1, save_best_only=True)

history = model.fit(train_data,
                    train_labels,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[checkpointer],
                    validation_data=(validation_data, validation_labels))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()
plt.savefig('Fig_2.png')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()
plt.savefig('Fig_3.png')

batch_size = 5
epochs = 5

input_tensor = Input(shape=(img_height, img_width, 3))

base_model = xception.Xception(weights='imagenet',
                               include_top=False,
                               input_shape=(img_width, img_height, 3),
                               pooling='avg')

top_model = Sequential()

top_model.add(Dense(256, activation='relu',
                    input_shape=base_model.output_shape[1:]))
top_model.add(Dropout(0.5))

top_model.add(Dense(128, activation='relu'))
top_model.add(Dropout(0.5))

top_model.add(Dense(1, activation='sigmoid'))

top_model.load_weights('top-weights.h5')

model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

model.compile(optimizer=SGD(lr=0.005, momentum=0.1, nesterov=True),
              loss='binary_crossentropy',
              metrics=['accuracy'])


checkpointer = ModelCheckpoint(
    filepath='weights.h5', verbose=1, save_best_only=True)

train_datagen = image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# model.fit
history = model.fit(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size)

test_generator = data_generator.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)

# score = model.evaluate_generator(test_generator, batch_size)
score = model.evaluate(test_generator, batch_size)
model.save("model.h5")
print("Loss: ", score[0], "Accuracy: ", score[1])

