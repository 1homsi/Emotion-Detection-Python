# first neural network with keras make predictions
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

def neuralNetwork():
	dataset = pd.read_csv("ML_methods\Dataset\pima-indians-diabetes.csv")

	# define the keras model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# compile the keras model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit the keras model on the dataset
	model.fit(dataset, dataset, epochs=150, batch_size=10, verbose=0)
	# make class predictions with the model
	predictions = (model.predict(dataset) > 0.5).astype(int)
	# summarize the first 5 cases
	for i in range(5):
		print('%s => %d (expected %d)' % (dataset.to_string(), predictions[i], dataset))