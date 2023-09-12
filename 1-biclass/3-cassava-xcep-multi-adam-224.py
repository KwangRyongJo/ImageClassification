# Fine-tuning
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
# from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import MaxPooling2D
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from test1 import no_of_classes, X_train, y_train, X_valid, y_valid, X_test, y_test, labels


# fix seed for reproducible results (only works on CPU, not GPU)
# seed = 9
# np.random.seed(seed=seed)
# tf.set_random_seed(seed=seed)


# load the Xception model without the final layers(include_top=False)

IMAGE_SIZE = [224, 224]
base_model = tf.keras.applications.Xception(
    input_shape=[*IMAGE_SIZE, 3], include_top=False)
print('Loaded model!')

# Let's freeze the first 15 layers - if you see the VGG model layers below,
# we are freezing till the last Conv layer.
for layer in base_model.layers[:15]:
    layer.trainable = False
base_model.summary()

# Now, let's create a top_model to put on top of the base model(we are not freezing any layers of this model)
top_model = Sequential()
# top_model.add(MaxPooling2D((2, 2)))
top_model.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:]))
Dense(2048, activation='relu', name='dense_zero')
Dense(1024, activation='relu', name='dense_one')
Dense(1024, activation='relu', name='dense_two')
Dense(1024, activation='relu', name='dense_three')
Dense(1024, activation='relu', name='dense_four')
Dense(512, activation='relu', name='dense_five')
top_model.add(Dense(no_of_classes, activation='softmax'))
top_model.summary()


# Let's build the final model where we add the top_model on top of base_model.
model = Sequential()
model.add(base_model)
model.add(top_model)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              # optimizer=tf.keras.optimizers.Adam(lr=0.001),
              # optimizer=SGD(lr=0.001, momentum=0.9),
              metrics=['accuracy'])


# When we check the summary below,  and trainable params for model is 7,081,989 = 7,079,424 + 2,565
# Time to train our model !
epochs = 15
batch_size = 8
best_model_finetuned_path = 'model-xcep-multi-adam-224-8.h5'

train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(
    X_train, y_train,
    batch_size=batch_size)

validation_generator = test_datagen.flow(
    X_valid, y_valid,
    batch_size=batch_size)

checkpointer = ModelCheckpoint(
    best_model_finetuned_path, save_best_only=True, verbose=1)

history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(X_valid) // batch_size,
    callbacks=[checkpointer])


model.load_weights(best_model_finetuned_path)

(eval_loss, eval_accuracy) = model.evaluate(
    X_test, y_test, batch_size=batch_size, verbose=1)

print("Accuracy: {:.2f}%".format(eval_accuracy * 100))
print("Loss: {}".format(eval_loss))

# Let's visualize some random test prediction.


def visualize_pred(y_pred):
    # plot a random sample of test images, their predicted labels, and ground truth
    fig = plt.figure(figsize=(16, 9))

    for i, idx in enumerate(np.random.choice(X_test.shape[0], size=16, replace=False)):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(X_test[idx]))
        pred_idx = np.argmax(y_pred[idx])
        true_idx = np.argmax(y_test[idx])
        ax.set_title("{} ({})".format(labels[pred_idx], labels[true_idx]),
                     color=("green" if pred_idx == true_idx else "red"))


visualize_pred(model.predict(X_test))
plt.savefig('Fig_4.png')

# Let's visualize the loss and accuracy wrt epochs


def plot(history):
    plt.figure(1)

    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig('Fig_5.png')


plot(history)
