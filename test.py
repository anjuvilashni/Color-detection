from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train = X_train / 255.0
X_test = X_test / 255.0
# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]
# Create the model
model = Sequential()
model.add(
    Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation="relu", padding="same")
)
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation="relu", kernel_constraint=MaxNorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation="relu", kernel_constraint=MaxNorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation="softmax"))

model.save("model.h5")
model.load('model.h5')
IMG_SIZE = 32
test_image = cv2.resize(cv2.imread("horse.jpg", cv2.IMREAD_GRAYSCALE),  (IMG_SIZE,IMG_SIZE))

test_image = np.array(test_image).reshape( -1, IMG_SIZE, IMG_SIZE, 1)

prediction = model.predict({'input': test_image })

print(prediction)