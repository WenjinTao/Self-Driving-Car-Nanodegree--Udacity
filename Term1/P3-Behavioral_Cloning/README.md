# Self-Driving Car Nano-degree

## Project 3: Behavioral Cloning

### 1. Introduction

This project is to train a CNN model to autonomously drive a car in the simulator using behavioral cloning method. 

The simulator is provided by Udacity, which has two playing modes: training mode and autonomous mode. Training mode is to drive the car by user and collect the driving behavioral data at the same time. Autonomous mode is to autonomously drive the car using the created CNN model and learned parameters.

![](figures/simulator.PNG)
### 2. Data

#### 2.1 Collecting Training Data

The training data can be collected via the simulator in the training mode. Also Udacity provides us a sample data for track 1, which was used in my project.

#### 2.2 Data Pre-processing

The pre-processing steps used in this project are listed below:

* To make the model lightweight and accelerate the training process, the original images were resized from 320x160 to 32x16.

* The color space was converted from RGB to HSV.

* S channel was selected for training.



**Recovery**

Multiple cameras were used to recover the car's direction from being off-center. The images from left and right cameras were appended to the training data, their corresponding steering angles were biased from the original angles, _+delta/-delta_, respectively.

**Save the Data** 
The processed data was saved as 'X.data.npy' and 'y.data.npy' for the future training process.


### 3. Modeling

As reference, I list the NVDIA and Comma.ai CNN architectures in the following sections. After that, the architecture I used for this project will be explained in 3.3.

#### 3.1 NVIDIA Model

The NVIDIA model can be found at [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). It was implemented in Keras, which is listed below.

````python
# NVIDIA Model
# Create the Sequential model
model = Sequential()
input_shape = (66, 200, 3)
# Layer Normalization - Add a lambda layer
model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=input_shape))
# Layer Conv.1
model.add(Conv2D(24, 5, 5,  subsample=(2,2), border_mode='valid',activation='relu'))
# Layer Conv.2
model.add(Conv2D(36, 5, 5, subsample=(2,2), border_mode='valid',activation='relu'))
# Layer Conv.3
model.add(Conv2D(48, 5, 5, subsample=(2,2), border_mode='valid',activation='relu'))
# Layer Conv.4
model.add(Conv2D(64, 3, 3, border_mode='valid',activation='relu'))
# Layer Conv.5
model.add(Conv2D(64, 3, 3, border_mode='valid',activation='relu'))
# Layer Flatten - Add a flatten layer
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
````
#### 3.2 Comma.ai Model

The Comma.ai model can be found at [train_steering_model ](https://github.com/commaai/research/blob/master/train_steering_model.py).  

````python
# Comma.ai model
model = Sequential()
input_shape = (160, 320, 3)
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape)
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))
````

#### 3.3 Model Used in this Project

The model I used in this project has a relatively lightweight architecture. It has 7 layers as listed below:

1. Normalization layer
2. 1st Convolutional layer
3. Maxpooling layer
4. Dropout layer
5. 2nd Convolutional layer
6. Flatten layer
7. Dense layer with the steering angle output

````python
# CNN model
model = Sequential()
input_shape = (16, 32, 1)
model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=input_shape))
model.add(Conv2D(4, 3, 3, border_mode='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Dropout(0.25))
model.add(Conv2D(16, 3, 3))
model.add(Flatten())
model.add(Dense(1))
````
**model.summary:**

| Layer (type)                    | Output Shape      | Param # | Connected to          |
| ------------------------------- | ----------------- | ------- | --------------------- |
| lambda_1 (Lambda)               | (None, 16, 32, 1) | 0       | lambda_input_1[0][0]  |
| convolution2d_1 (Convolution2D) | (None, 14, 30, 4) | 40      | lambda_1[0][0]        |
| maxpooling2d_1 (MaxPooling2D)   | (None, 3, 7, 4)   | 0       | convolution2d_1[0][0] |
| dropout_1 (Dropout)             | (None, 3, 7, 4)   | 0       | maxpooling2d_1[0][0]  |
| convolution2d_2 (Convolution2D) | (None, 1, 5, 16)  | 592     | dropout_1[0][0]       |
| flatten_1 (Flatten)             | (None, 80)        | 0       | convolution2d_2[0][0] |
| dense_1 (Dense)                 | (None, 1)         | 81      | flatten_1[0][0]       |

Total params: 713
Trainable params: 713
Non-trainable params: 0

### 4. Training

The model is trained to minimize the mean-squared error using 'adam' optimizer, with a batch size of 256 and epoch of 15. 20% of the data was used for validation.

````python
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X, y, batch_size=256, nb_epoch=15, verbose=1, validation_split=0.2)
````

**Validation & Evaluation**

This model was validated by applying the learned parameters onto the validation dataset. After many experiments, loss~0.025 is an acceptable criterion.

Finally the model was evaluated on both of the tracks in the simulator.

### 5. Summary

The S channel of the HSV color space with image size of 32x16 provides us enough information to train our model. This created CNN model has 1 Lambda layer to normalize the input, 2 convolutional layers to add its depth, 1 maxpooling layer to pick the significant features, 1 dropout layer to avoid overfitting, 1 flatten layer with 1 dense layer to obtain the steering angel value.

After training, this model successfully drove the car on both of the tracks by itself.

### 6. Appendix

#### 6.1 File Structure

````html
+-- data/
|	+-- IMG/
|	|	+-- center_2016_12_01_13_30_48_287.jpg
|	|	...
|	+-- driving_log.csv
+-- X.data.npy - The training data
+-- y.data.npy - The training data
+-- model.py - The script used to create and train the model.
+-- drive.py - The script to drive the car.
+-- model.json - The model architecture saved from Keras.
+-- model.h5 - The model weights saved from Keras.
+-- figures/
|	+-- simulator.PNG
|	...
````


#### 6.2 Videos of Evaluation

Video of the CNN performance on the two tracks can be found here:

Track 1:
[![Track 1](https://img.youtube.com/vi/VcyPwxqqN5E/0.jpg)](https://www.youtube.com/watch?v=VcyPwxqqN5E)
Track 2:
[![Track 2](https://img.youtube.com/vi/f8TJ_bV4DKA/0.jpg)](https://www.youtube.com/watch?v=f8TJ_bV4DKA)
#### 6.3 Requirements

* anaconda
* tensorflow
* keras
* cv2
* To be added if needed...
