from tensorflow.keras.datasets import cifar10

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
from PIL import Image

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)


def cnn_model():
    input_shape = (32, 32, 3)
    n_classes = 10
    model = Sequential()

    # First Conv layer
    model.add(
        Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu",
            kernel_constraint=MaxNorm(1e-7),
            input_shape=input_shape,
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Second Conv layer
    model.add(
        Conv2D(
            256,
            (3, 3),
            padding="same",
            activation="relu",
            kernel_constraint=MaxNorm(1e-7),
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Third, fourth, fifth convolution layer
    model.add(
        Conv2D(
            512,
            (3, 3),
            padding="same",
            activation="relu",
            kernel_constraint=MaxNorm(1e-7),
        )
    )
    model.add(
        Conv2D(
            512,
            (3, 3),
            padding="same",
            activation="relu",
            kernel_constraint=MaxNorm(1e-7),
        )
    )
    model.add(
        Conv2D(
            256,
            (3, 3),
            padding="same",
            activation="relu",
            kernel_constraint=MaxNorm(1e-7),
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Fully Connected layers
    model.add(Flatten())

    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(n_classes, activation="softmax"))

    model.summary()

    return model


datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
)

datagen.fit(X_train)

model = cnn_model()
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.0003, decay=1e-6),
    metrics=["accuracy"],
)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    steps_per_epoch=len(X_train) // 64,
    epochs=25,
    validation_data=(X_valid, y_valid),
    verbose=1,
)

# Plotting the train and val accuracy and loss
pd.DataFrame(history.history).plot()

# Evaluating model on the test set
scores = model.evaluate(X_test, y_test)

# Make predictions
pred = model.predict(X_test)
labels = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]
y_pred = np.argmax(pred, axis=1)
y_true = np.argmax(y_test, axis=1)
errors = y_pred - y_true != 0

# Print Classification Report
print(classification_report(y_true, y_pred))

# Check the predictions
fig, axes = plt.subplots(5, 5, figsize=(12, 12))
axes = axes.ravel()

for i in np.arange(25):
    axes[i].imshow(X_test[i])
    axes[i].set_title("True: %s \nPredict: %s" % (labels[y_true[i]], labels[y_pred[i]]))
    axes[i].axis("off")
    plt.subplots_adjust(wspace=1)

# Check the wrong predictions
fig, axes = plt.subplots(5, 5, figsize=(12, 12))
axes = axes.ravel()

miss_pred = np.where(y_pred != y_true)[0]
for i in np.arange(25):
    axes[i].imshow(X_test[miss_pred[i]])
    axes[i].set_title(
        "True: %s \nPredict: %s"
        % (labels[y_true[miss_pred[i]]], labels[y_pred[miss_pred[i]]])
    )
    axes[i].axis("off")
    plt.subplots_adjust(wspace=1)

model.save("cifar10_cnn.h5")
