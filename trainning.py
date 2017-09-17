import csv
import cv2
import numpy as np

def read_image(source_path):
	#print(source_path)
	filename = source_path.split('/')[-1]
	#print(filename)
	filename = './data/IMG/'+ filename.split('\\')[-1]
	#print(filename)
	image = cv2.imread(filename)
	return image

lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	steering_center = float(line[3])
	# create adjusted steering measurements for the side camera images
	correction = 0.2 # this is a parameter to tune
	steering_left = steering_center + correction
	steering_right = steering_center - correction
   
	# Center
	images.append(read_image(line[0]))
	measurements.append(steering_center)
	# Left
	images.append(read_image(line[1]))
	measurements.append(steering_left)
	# Right
	images.append(read_image(line[2]))
	measurements.append(steering_right)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image, flipCode=1))
	augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Cropping2D

model  = Sequential()
# Normalize and mean centering the data 
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)



model.save('model.h5')
