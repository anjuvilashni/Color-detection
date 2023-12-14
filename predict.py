from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np

dict = [
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

IMG_SIZE = 32
image = cv2.imread("upload/test.jpg")
image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
image = np.array(image).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train / 255.0
X_test = X_test / 255.0
# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]


def classify():
    INPUT_SHAPE = (32, 32, 3)
    KERNEL_SIZE = (3, 3)
    model = Sequential()
    # Convolutional Layer
    model.add(
        Conv2D(
            32,
            KERNEL_SIZE,
            input_shape=INPUT_SHAPE,
            activation="relu",
            padding="same",
        )
    )
    model.add(BatchNormalization())
    model.add(
        Conv2D(
            32,
            KERNEL_SIZE,
            activation="relu",
            padding="same",
        )
    )
    model.add(BatchNormalization())
    # Pooling layer
    model.add(MaxPool2D(pool_size=(2, 2)))
    # Dropout layers
    model.add(Dropout(0.25))
    model.add(
        Conv2D(
            64,
            KERNEL_SIZE,
            activation="relu",
            padding="same",
        )
    )
    model.add(BatchNormalization())
    model.add(
        Conv2D(
            64,
            KERNEL_SIZE,
            activation="relu",
            padding="same",
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(
        Conv2D(
            128,
            KERNEL_SIZE,
            activation="relu",
            padding="same",
        )
    )

    model.add(BatchNormalization())
    model.add(
        Conv2D(
            128,
            KERNEL_SIZE,
            activation="relu",
            padding="same",
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation="softmax"))

    model.load_weights("model.h5")
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    predicted_class_label = dict[predicted_class]
    return predicted_class_label


if __name__ == "__main__":
    classify()
