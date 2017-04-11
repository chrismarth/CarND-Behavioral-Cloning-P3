import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import cv2
import csv

# Read each sample from the CSV data acquisition file
data_prefix = 'data/baseline_2laps/'
samples = []
with open(data_prefix + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = data_prefix + 'IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Setup model
model = Sequential()

# Pre-processing
# 1. TODO: Trim top half of image
# 2. Convert to grayscale
# 3. Normalize and center between -1 and 1
model.add(Lambda(lambda x: tf.image.rgb_to_grayscale(x), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(160,320,1)))

# Network setup
model.add(Flatten(input_shape=(160,320,1)))
model.add(Dense(1))

# Run backpropagation
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples),
                    validation_data=validation_generator, nb_val_samples=len(validation_samples),
                    nb_epoch=3)
model.save('model.h5')
