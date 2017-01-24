import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


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


def X_prep(X, driving_log):
    """
    Append the data of left and right camera to the original data of the center camera
    :param X: A place holder defining the data shape
    :param driving_log: The driving log loaded from file
    :return: Augmented X
    """
    
    (img_center_names, img_left_names, img_right_names) = (driving_log['center'].values,
                                                           driving_log['left'].values,
                                                           driving_log['right'].values)
    
    for img_center_name, img_left_name, img_right_name in zip(img_center_names, img_left_names, img_right_names):
        img_center = plt.imread('data/'+img_center_name)
        img_left = plt.imread('data/'+img_left_name.lstrip())
        img_right = plt.imread('data/'+img_right_name.lstrip())

        # Image pre-processing
        img_center = img_prep(img_center)
        img_left = img_prep(img_left)
        img_right = img_prep(img_right)

        X = np.append(X, [np.expand_dims(img_center, 3)], axis=0)
        X = np.append(X, [np.expand_dims(img_left, 3)], axis=0)
        X = np.append(X, [np.expand_dims(img_right, 3)], axis=0)
    
    return X


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


if __name__ == '__main__':

    # Load the driving log data
    driving_log = pd.read_csv("data/driving_log.csv")

    # Define the shape of X
    X = np.empty([0, 16, 32, 1])
    # Load the steering angle
    y = driving_log['steering'].values

    # Prepare the data from central/ left/ right cameras
    X = X_prep(X, driving_log)
    y = y_prep(y, delta=.25)

    # Flip the images horizontally to double the training data
    X = np.append(X, X[:, :, ::-1, :], axis=0)
    y = np.append(y, -y)

    print("Shapes of the training data: X-{}, y-{}".format(X.shape, y.shape))

    # num of Neg. steering, Pos. steering, Zero steering
    y_neg = y[y < -0.0]
    y_pos = y[y > 0.0]
    y_zero = y[y == 0.0]
    print("Number of Neg.{}/ Pos.{}/ Zero.{} steering.".format(y_neg.shape[0], y_pos.shape[0], y_zero.shape[0]))

    # Save the Data for convenience
    np.save('X_025.data', X)
    np.save('y_025.data', y)
    print('Saved data to disk')
