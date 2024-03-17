# We are creating easiest neural network using TensorFlow
# I have made this model only to predict that the given input from user have a difference of 10 units or not.  
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

model = Sequential([
    Dense(units=5, activation="sigmoid"), #first layer,  5 neurons with sigmoid activation function
    Dense(units=3, activation="sigmoid"), #second layer, 3 neurons with sigmoid activartion function
    Dense(units=1, activation="sigmoid")]) # In third layer, we are using activation function as Sigmoid, that's not a good practise. 

x = np.array([[20.0, 15.0],
              [30.0, 20.0],
              [40.0, 25.0],
              [50.0, 40.0],
              [20.0, 10.0]])

y = np.array([0.0, 1.0, 0.0, 1.0, 1.0]) # Target values repect to each input data from x

model.compile(optimizer='adam', loss='mse')  # Compiling the model with optimizer and loss function
model.fit(x, y, epochs=1000, verbose=0)  # Training the model for 1000 epochs(It will run for 1000 times for given dataset x)
yhat = model.predict([[60.0, 50.0]]).flatten()  # Predicting for the values [60.0, 50.0], In place of these values give your own for understanding. 

if yhat >= 0.5:
    print(yhat, "Yes, The given values have a postive difference of 10 Units or more.")
else:
    print(yhat, "No, The given values do not have a positive difference of 10 Units or more.")