import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Lambda, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2, activity_l2


def cnn(input_shape):
    """
    Build up the CNN architecture
    :param input_shape:
    :return: the CNN model
    """
    model = Sequential()
    # Lambda layer to implement the normalization
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=input_shape))
    model.add(Conv2D(4, 3, 3, border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.25))
    model.add(Conv2D(16, 3, 3))
    model.add(Flatten())
    model.add(Dense(1))
    return model

if __name__ == '__main__':

    # Load the training data
    X = np.load('X_025.data.npy')
    y = np.load('y_025.data.npy')

    # Shuffling
    X, y = shuffle(X, y)

    # Build up the CNN
    input_shape = (16, 32, 1)
    model = cnn(input_shape)
    model.summary()

    # Train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X, y, batch_size=256, nb_epoch=15, verbose=1, validation_split=0.2)

    # Save the model
    model_json = model.to_json()
    with open("model_lightweight.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("model_lightweight.h5")
    print("Saved model to disk")
