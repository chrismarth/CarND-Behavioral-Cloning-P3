import numpy as np
from keras.models import Sequential
from keras.layers import Cropping2D, Dense, Flatten, Lambda, Conv2D
from keras.optimizers import Adam
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import csv
import math


def generator(samples, steer_correction=0.2, batch_size=32):
    """
    Given a set of samples, steer correction for left/right cameras, and a batch size, return an iterator that returns
    a 2-tuple of numpy arrays with the first value in the tuple being the input image and the second value the steer angle
    
    :param samples: The total number of test samples
    :param steer_correction: The steering correction to add to the left camera or subtract from the right camera
    :param batch_size: The size of the sample batch that the iterator returns
    :return: An iterator that on iteration returns a 2-tuple of numpy arrays containing input data and steer angle
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = batch_sample[7] + 'IMG/' + batch_sample[0].split('/')[-1]
                left_name = batch_sample[7] + 'IMG/' + batch_sample[1].split('/')[-1]
                right_name = batch_sample[7] + 'IMG/' + batch_sample[2].split('/')[-1]
                center_image = cv2.cvtColor(cv2.imread(center_name), cv2.COLOR_BGR2RGB)
                left_image = cv2.cvtColor(cv2.imread(left_name), cv2.COLOR_BGR2RGB)
                right_image = cv2.cvtColor(cv2.imread(right_name), cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                left_angle = center_angle + steer_correction
                right_angle = center_angle - steer_correction
                images.extend([center_image, left_image, right_image])
                angles.extend([center_angle, left_angle, right_angle])
                if batch_sample[8]:                                     # Should we flip the images?
                    center_image_flipped = np.fliplr(center_image)
                    left_image_flipped = np.fliplr(left_image)
                    right_image_flipped = np.fliplr(right_image)
                    center_angle_flipped = -center_angle
                    left_angle_flipped = -left_angle
                    right_angle_flipped = -right_angle
                    images.extend([center_image_flipped, left_image_flipped, right_image_flipped])
                    angles.extend([center_angle_flipped, left_angle_flipped, right_angle_flipped])
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# Create the complete set of test samples from a set of CSV data acquisition files.
# Each sample is a tuple. The first element is the directory of images and the second is True/False if we should flip
test_cases = [('data/baseline_2laps/', True), ('data/extra_corners/', True), ('data/recovery/', True), ('data/recovery2/', False), ('data/recovery3/', False)]
test_samples = []
for test_case in test_cases:
    test_case_dir = test_case[0]
    test_case_flip = test_case[1]
    with open(test_case_dir + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            line.append(test_case_dir)
            line.append(test_case_flip)
            test_samples.append(line)

# Create training and validation sets and create generators that iterates through each sets
train_samples, validation_samples = train_test_split(test_samples, test_size=0.2)
train_generator = generator(train_samples, steer_correction=0.5, batch_size=32)
validation_generator = generator(validation_samples, steer_correction=0.5, batch_size=32)

# Setup model
model = Sequential()

# Pre-processing Layers
model.add(Cropping2D(cropping=((60, 0), (0, 0)), input_shape=(160, 320, 3)))    # Crops the top 70 pixels
model.add(Lambda(lambda x: (x / 127.5) - 1.0))                                  # Normalize and center channel data

# Network setup - meant to mimic the Nvidia architecture
model.add(Conv2D(24, 5, strides=2, activation="relu"))
model.add(Conv2D(36, 5, strides=2, activation="relu"))
model.add(Conv2D(48, 5, strides=2, activation="relu"))
model.add(Conv2D(64, 3, activation="relu"))
model.add(Conv2D(64, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Run backpropagation
adam = Adam(lr=0.001)   # In case we need to tune the learning rate - we didn't need to do this
model.compile(loss='mse', optimizer=adam)      # Use Mean-Squared Error and the ADAM optimizer
model.fit_generator(train_generator, steps_per_epoch=math.floor(len(train_samples)/32), validation_data=validation_generator, validation_steps=math.floor(len(validation_samples)/32), epochs=5)

# Save the model so we can use it drive the vehicle
model.save('model.h5')

# This fixes a bug in Tensorflow related to closing the session
K.clear_session()
