# Self-Driving Car Nanodegree

## Project 3: Behavioral Cloning

### Overview

This project is to train a CNN model to autonomously drive a car in the simulator using behavioral cloning method. 

The simulator is provided by Udacity, which has two playing modes: training mode and autonomous mode. Training mode is to drive the car by user and collecting the driving behavioral data at the same time. Autonomous mode is to autonomously drive the car using the created CNN model and learned parameters.

![](figures/simulator.png)



### Modeling

.

````python
# Create the Sequential model
model = Sequential()
input_shape = (16, 32, 1)

# Layer Normalization - Add a lambda layer
model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=input_shape))

model.add(Conv2D(4, 3, 3, border_mode='valid', activation='relu'))

model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Dropout(0.25))
model.add(Conv2D(16, 3, 3))

model.add(Flatten())

model.add(Dense(1))
````



### Training



````python
# Optimizer
adam = Adam()

model.compile(optimizer=adam, loss='mean_squared_error')

history = model.fit(X, y, batch_size=256, nb_epoch=15, verbose=1, validation_split=0.2)
````



#### Collecting Training Data

#### Validation

This model was validated by applying the learned parameters onto the validation dataset.

Finally the model was evaluated on both of the tracks in the simulator.

### Files

data.py - The script used to preprocess the training data.

model.py - The script used to create and train the model.

drive.py - The script to drive the car.

model.json - The model architecture saved from Keras.

model.h5 - The model weights saved from Keras.



### Summary



### Video

Video of the CNN performance on the two tracks can be found here: [Track 1](), [Track 2]().