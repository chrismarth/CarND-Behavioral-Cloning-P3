import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import csv


def generator(samples, steer_correction=0.2, batch_size=32):
    """
    Given a set of samples and a batch size, return an iterator that returns a 2-tuple of numpy arrays with the first
    value in the tuple being the input image and the second value the label
    
    :param samples: The total number of test samples
    :param steer_correction: 
    :param batch_size: The size of the sample batch that the iterator will return
    :return: An iterator that on iteration returns a 2-tuple of numpy arrays containing input data and label
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
                center_image = cv2.cvtColor(cv2.imread(center_name), cv2.COLOR_BGR2RGB)
                left_image = cv2.cvtColor(cv2.imread(center_name), cv2.COLOR_BGR2RGB)
                right_image = cv2.cvtColor(cv2.imread(center_name), cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                left_angle = center_angle + steer_correction
                right_angle = center_angle - steer_correction
                images.extend(center_image, left_image, right_image)
                angles.extend(center_angle, left_angle, right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# Create the complete set of test samples from the set of CSV data acquisition files
test_cases = ['data/baseline_2laps/']
test_samples = []
for test_case in test_cases:
    with open(test_case + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            line.append(test_case)
            test_samples.append(line)

# Create training and validation sets and create generators that iterates through each sets
train_samples, validation_samples = train_test_split(test_samples, test_size=0.2)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Setup model
model = Sequential()

# Pre-processing Layers
# 1. TODO: Trim top half of image - eliminate the top 60 pixels since this is just background noise
# 2. TODO: Possibly downsample to a smaller image size
model.add(Lambda(lambda x: tf.image.rgb_to_grayscale(x), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(160, 320, 1)))

# Network setup
model.add(Flatten(input_shape=(160, 320, 1)))
model.add(Dense(1))

# Run backpropagation
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples),
                    validation_data=validation_generator, nb_val_samples=len(validation_samples),
                    nb_epoch=5)

# Save the model so we can use it drive the vehicle
model.save('model.h5')
