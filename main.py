import tensorflow as tf
import tensorflow_datasets as tfds
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import string, random
import pandas as pd

#https://medium.com/@ashraf.dasa/bean-disease-classification-using-tensorflow-convolutional-neural-network-cnn-2079dffe87ce

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

prjectName = 'beans'


def retrive_data():
    test_ds, info = tfds.load(prjectName, split='test', as_supervised=True, with_info=True, shuffle_files=True)
    print(info)
    # to see labels
    print(f'Classes:{info.features["label"].names}')
    # show the shape
    print(test_ds.element_spec)
    return test_ds, info


def get_training_data():
    validation_data = tfds.load(prjectName, split=f'validation', as_supervised=True)
    training_data = tfds.load(prjectName, split=f'train', as_supervised=True)
    return training_data, validation_data


def wrangle_data_GenPlus(dataset, split, batch_size=32):
    wrangled = dataset.map(lambda img, lbl: (tf.cast(img, tf.float32) / 255.0, lbl))
    if split == 'train':
        features = np.array([x[0] for x in wrangled])
        lables = np.array([x[1] for x in wrangled])
        train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            horizontal_flip=True,
            zoom_range=0.2,
            rotation_range=20,
            fill_mode='nearest'
        )
        wrangled = train_data_gen.flow(features, lables, batch_size=batch_size)
    elif split in ('valid', 'test'):  # Caches the elements in this dataset. loat it into the memory to go faster
        wrangled = wrangled.cache()
        wrangled = wrangled.batch(batch_size)  # Combines consecutive elements of this dataset into batches.
        wrangled = wrangled.prefetch(tf.data.AUTOTUNE)
    return wrangled


def compileModel(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  # classification with integer encoded labels use "scce"
                  metrics=['accuracy'])
    print(model.summary())
    return model


def myFullCNN():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer((500, 500, 3)),
        tf.keras.layers.experimental.preprocessing.Resizing(125, 125),
        tf.keras.layers.Conv2D(64, 3, activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, 3, activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(128, 3, activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(128, 3, activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(128, 3, activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(3, activation=tf.keras.activations.softmax)
    ], name='cnn_model')
    return compileModel(model)

num_filters = 15
filter_size = 10
pool_size = 4
strides = 2
fc_output = 128
drop_probability = 0.25
learning_rate = 0.001
image_height = 500
image_width = 500
num_channels = 3  # RGB
num_classes = 3  # healthy, angular leaf spot disease, bean rust disease

# Pipeline hyperparameters:
batch_size = 32
def model2():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(num_filters, filter_size, input_shape=(image_height, image_width, 3),
                               strides=strides, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=pool_size),
        tf.keras.layers.Dropout(drop_probability),
        tf.keras.layers.Conv2D(num_filters, filter_size, strides=strides, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=pool_size),
        tf.keras.layers.Dropout(drop_probability),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(fc_output, activation='relu'),

        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])
    return compileModel(model)

def plot_History(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


if __name__ == "__main__":
    # prepare the data
    test_ds, info = retrive_data()
    train_ds, valid_ds = get_training_data()
    batch_size = 64
    train_data = wrangle_data_GenPlus(train_ds, 'train', batch_size=batch_size)
    valid_data = wrangle_data_GenPlus(valid_ds, 'valid', batch_size=batch_size)
    test_data = wrangle_data_GenPlus(test_ds, 'test', batch_size=batch_size)

    # advancedCNN
    model = myFullCNN()

    # fit the model
    history = model.fit(train_data, validation_data=valid_data, epochs=50)

    plot_History(history)
    print(model.evaluate(test_data))

    plot_History(history)
