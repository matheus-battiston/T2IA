import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from keras import backend as K

# https://medium.com/@ashraf.dasa/bean-disease-classification-using-tensorflow-convolutional-neural-network-cnn-2079dffe87ce
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
checkpoint_dir = './training_checkpoints'

prjectName = 'beans'


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))


def recall(y_true, y_pred):
    # recall of class 1

    # do not use "round" here if you're going to use this as a loss function
    true_positives = K.sum(K.round(y_pred) * y_true)
    possible_positives = K.sum(y_true)
    return true_positives / (possible_positives + K.epsilon())


lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=33 * 1000,
    decay_rate=1,
    staircase=False)


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(y_pred) * y_true)
    predicted_positives = K.sum(y_pred)
    precision_keras = true_positives / (predicted_positives + K.epsilon())
    return precision_keras


def resize_and_rescale(image, label):
    image = tf.image.resize(image, [tam, tam])
    return image, label


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
    if split:
        features = np.array([x[0] for x in wrangled])
        lables = np.array([x[1] for x in wrangled])
        train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            horizontal_flip=True,
            zoom_range=0.2,
            rotation_range=20,
            fill_mode='nearest'
        )
        wrangled = train_data_gen.flow(features, lables, batch_size=batch_size)
    else:  # Caches the elements in this dataset. loat it into the memory to go faster
        wrangled = wrangled.cache()
        wrangled = wrangled.batch(batch_size)  # Combines consecutive elements of this dataset into batches.
        wrangled = wrangled.prefetch(tf.data.AUTOTUNE)

    return wrangled


def compileModel(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy', recall, precision, f1])
    return model


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
tam = 224

# Pipeline hyperparameters:
batch_size = 32



def model1():
    neural_net = tf.keras.Sequential([
        mobile_net_layers,
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return compileModel(neural_net)


def model2():
    neural_net = tf.keras.Sequential([
        inception_layers,
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return compileModel(neural_net)


def model3():
    neural_net = tf.keras.Sequential([
        resnet_layers,
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return compileModel(neural_net)


def model4():
    neural_net = tf.keras.Sequential([
        inception_resnet_layers,
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return compileModel(neural_net)

def model5():
    neural_net = tf.keras.Sequential([
        inaturalist_layers,
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return compileModel(neural_net)

def plot_History(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


def exec_mobile_net():
    epochs_range = range(50)
    model = model1()
    history = model.fit(train_ds, validation_data=valid_ds, epochs=50)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    plot_History(history)
    print(model.evaluate(test_ds))


def exec_inception():
    epochs_range = range(50)
    model = model2()
    history = model.fit(train_ds, validation_data=valid_ds, epochs=50)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    plot_History(history)
    print(model.evaluate(test_ds))


def exec_resnet():
    epochs_range = range(50)
    model = model3()
    history = model.fit(train_ds, validation_data=valid_ds, epochs=50)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    plot_History(history)
    print(model.evaluate(test_ds))


def exec_inception_resnet():
    epochs_range = range(50)
    model = model4()
    history = model.fit(train_ds, validation_data=valid_ds, epochs=50)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    plot_History(history)
    print(model.evaluate(test_ds))

def exec_inaturalist():
    epochs_range = range(50)
    model = model5()
    history = model.fit(train_ds, validation_data=valid_ds, epochs=50)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    plot_History(history)
    print(model.evaluate(test_ds))



if __name__ == "__main__":
    # prepare the data

    mobilenet_v3 = "https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/classification/5"
    mobile_net_layers = hub.KerasLayer(mobilenet_v3, input_shape=(tam, tam, 3))
    mobile_net_layers.trainable = False

    inception = "https://tfhub.dev/google/tf2-preview/inception_v3/classification/4"
    inception_layers = hub.KerasLayer(inception, input_shape=(tam, tam, 3))
    inception_layers.trainable = False

    resnet = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"
    resnet_layers = hub.KerasLayer(resnet, input_shape=(tam, tam, 3))
    resnet_layers.trainable = False

    inception_resnet = "https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/5"
    inception_resnet_layers = hub.KerasLayer(inception_resnet, input_shape=(tam, tam, 3))
    inception_resnet_layers.trainable = False

    inaturalist = "https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5"
    inaturalist_layers = hub.KerasLayer(inaturalist, input_shape=(tam, tam, 3))
    inaturalist_layers.trainable = False


    test_ds, info = retrive_data()
    train_ds, valid_ds = get_training_data()
    batch_size = 32
    train_ds = train_ds.map(resize_and_rescale, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    valid_ds = valid_ds.map(resize_and_rescale, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(resize_and_rescale, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = wrangle_data_GenPlus(train_ds, True, batch_size=batch_size)
    valid_ds = wrangle_data_GenPlus(valid_ds, False, batch_size=batch_size)
    test_ds = wrangle_data_GenPlus(test_ds, False, batch_size=batch_size)

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6)

    exec_mobile_net()
    # exec_inception()
    # exec_resnet()
    # exec_inception_resnet()
    # exec_inaturalist()
