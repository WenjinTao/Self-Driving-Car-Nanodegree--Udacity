import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Lambda, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2, activity_l2

def img_prep(img):
    """
    Pre-process the image
    :param img: The original image with RGB channels
    :return: A width:32, height:16 image with S channel in HSV mode
    """
    # Convert the color space
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 1]
    # Resize the image to W32H16
    img = cv2.resize(img, (32, 16))

    return img


def X_prep(driving_log):
    """
    Organize the names of the center, left and right cameras into one list for being used in the generator
    :param driving_log: The driving log loaded from file
    :return: X_names
    """
    X_names = []

    (img_center_names, img_left_names, img_right_names) = (driving_log['center'].values,
                                                           driving_log['left'].values,
                                                           driving_log['right'].values)

    for img_center_name, img_left_name, img_right_name in zip(img_center_names, img_left_names, img_right_names):
        X_names.append(img_center_name)
        X_names.append(img_left_name.lstrip())
        X_names.append(img_right_name.lstrip())

    return X_names


def y_prep(y, delta):
    """
    Prepare the y data
    :param y: original y data before left/right bias
    :param delta: Bias value
    :return: Biased y
    """
    y = np.repeat(y, 3)
    for i in np.arange(1, y.shape[0], 3):

        y[i] += delta
        y[i+1] -= delta

    return y


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


def batch_generator(X_names, y, batch_size=256):

    while 1:
        X_batch = np.zeros([batch_size, 16, 32, 1])
        y_batch = np.zeros([batch_size, 1])

        # Using batch_size/2 data first, then double it by horizontally flipping to form the full batch size
        for i in range(int(batch_size/2)):
            random_choice = int(np.random.choice(len(X_names),1))
            img_orig = plt.imread('data/'+X_names[random_choice])
            img = img_prep(img_orig)

            X_batch[2*i] = np.expand_dims(img, 3)
            y_batch[2*i] = y[random_choice]
            # Add the horizontally flipped data into this patch
            X_batch[2*i+1] = np.expand_dims(img[:, ::-1], 3)
            y_batch[2*i+1] = -y[random_choice]

        yield (X_batch, y_batch)


if __name__ == '__main__':

    # Load the driving log data
    driving_log = pd.read_csv("data/driving_log.csv")

    # Prepare the data from central/ left/ right cameras
    X_names = X_prep(driving_log)
    # Load and Prepare the steering angles
    y = driving_log['steering'].values
    y = y_prep(y, delta=.25)

    # Shuffling for splitting
    X_names, y = shuffle(X_names, y)

    # Splitting
    X_train_names, X_validation_names, y_train, y_validation = train_test_split(X_names, y, test_size=0.2, random_state=0)

    # Build up the CNN
    input_shape = (16, 32, 1)
    model = cnn(input_shape)
    model.summary()

    # Train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Using generator
    history = model.fit_generator(generator=batch_generator(X_train_names, y_train, batch_size=256),
                                  samples_per_epoch=38400,
                                  nb_epoch=15,
                                  verbose=1,
                                  validation_data=batch_generator(X_validation_names, y_validation, batch_size=256),
                                  nb_val_samples=9728)

    # Save the model
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("model.h5")
    print("Saved model to disk")
